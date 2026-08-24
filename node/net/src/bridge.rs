//! The trainer bridge — the consensus boundary as a socket.
//!
//! The node owns consensus and networking; training is an UNCONSTRAINED local
//! compute plugin (§6.3): a PyTorch process (client/miner_bridge.py) connects
//! on localhost, receives the head state once, then per round trains and
//! returns a COMPRESSED quantized delta. When the head advances, the node
//! sends the sparse state difference so the bridge stays synced without ever
//! re-shipping the full model.
//!
//! Frames: [u32 BE length][bytes]. Control messages are JSON; the one big blob
//! (the initial state) follows a {"bin_next": true} message as a raw frame of
//! i64-LE. Everything else (payloads, sparse advances) is small enough to ride
//! base64 inside JSON.

use crate::proto::{Payload, SparseI64};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{info, warn};

#[derive(Debug)]
pub enum ToBridge {
    /// v1: `experts_per_layer` lets the trainer build the exact (possibly
    /// grown, ragged) architecture before loading the chain-order state.
    State { height: u64, state: Vec<i64>, experts_per_layer: Vec<u64> },
    /// `budget_s`: how long the trainer may spend before its delta goes stale
    /// (derived from the node's block interval) — the trainer auto-fits its
    /// inner steps to it, so a slow GPU still lands includable deltas.
    /// v1: `min_nnz` is the consensus work quota's floor on nonzero
    /// coordinates (the compressor keeps at least this many), and
    /// `active_pages` is the claimable set — deltas must be zero on frozen
    /// pages or validation rejects them.
    Train { height: u64, seed: u64, budget_s: f64, min_nnz: u64,
            max_nnz: u64, quota_4dp: u64,
            active_pages: Vec<u32> },
    Advance { height: u64, dim: u64, sparse: SparseI64 },
    /// v1 GROWTH EVENT: the chain appended one expert page; the bridge appends
    /// the deterministic init to its state, instantiates the expert, and
    /// rebuilds its layout. `init` rides as a raw i64 frame after the JSON.
    Grow { height: u64, new_dim: u64, page_id: u64, layer: i64, expert: i64,
           init: Vec<i64> },
    Generate { prompt: String, n: u64 },
    /// rev 7: score candidate deltas on a held-out batch (seeded from block
    /// context). Each delta rides as a full-i64 sparse vector the trainer adds
    /// to its synced state, evaluates, and reverts.
    Eval { height: u64, seed: u64, deltas: Vec<(String, SparseI64)> },
}

#[derive(Debug)]
pub enum FromBridge {
    Connected,
    /// v1: `pages` is the trainer's claim set for this delta.
    Delta { height: u64, loss: f64, pages: Vec<u32>, payload: Payload },
    NeedState,
    Generated { text: String, height: u64 },
    /// rev 7: micro-nat held-out-loss improvements per txid.
    Scores { height: u64, scores: Vec<(String, u64)>,
             sketches: Vec<(String, Vec<i32>)> },
}

async fn write_frame<W: AsyncWriteExt + Unpin>(s: &mut W, bytes: &[u8]) -> std::io::Result<()> {
    s.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    s.write_all(bytes).await
}

async fn read_frame<R: AsyncReadExt + Unpin>(s: &mut R) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    s.read_exact(&mut len).await?;
    let n = u32::from_be_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf).await?;
    Ok(buf)
}

/// One bridge connection. The socket is SPLIT: a dedicated reader task drains
/// the trainer's frames no matter what — the node may be mid-write of a 20MB
/// state advance while the trainer is mid-write of an 18MB delta, and if one
/// task did both, the two large writes deadlock against full TCP buffers
/// (found live in the dress rehearsal).
async fn serve_one(
    sock: TcpStream,
    cmds: &mut mpsc::Receiver<ToBridge>,
    events: &mpsc::Sender<FromBridge>,
) -> std::io::Result<()> {
    let (mut rd, mut wr) = sock.into_split();
    // handshake: the bridge speaks first
    let hello: Value = serde_json::from_slice(&read_frame(&mut rd).await?)?;
    if hello["t"] != "hello" {
        return Ok(());
    }
    info!("trainer bridge connected");
    let _ = events.send(FromBridge::Connected).await;

    let ev = events.clone();
    let mut reader = tokio::spawn(async move {
        loop {
            let frame = read_frame(&mut rd).await?;
            let v: Value = serde_json::from_slice(&frame)?;
            match v["t"].as_str() {
                Some("delta") => {
                    let payload: Payload = serde_json::from_value(v["payload"].clone())
                        .map_err(std::io::Error::other)?;
                    let pages: Vec<u32> = v["pages"].as_array()
                        .map(|a| a.iter()
                            .filter_map(|x| x.as_u64().map(|p| p as u32))
                            .collect())
                        .unwrap_or_default();
                    let _ = ev.send(FromBridge::Delta {
                        height: v["height"].as_u64().unwrap_or(0),
                        loss: v["loss"].as_f64().unwrap_or(0.0),
                        pages,
                        payload,
                    }).await;
                }
                Some("resync") => {
                    let _ = ev.send(FromBridge::NeedState).await;
                }
                Some("generated") => {
                    let _ = ev.send(FromBridge::Generated {
                        text: v["text"].as_str().unwrap_or("").to_string(),
                        height: v["height"].as_u64().unwrap_or(0),
                    }).await;
                }
                Some("scores") => {
                    let scores = v["scores"].as_object()
                        .map(|m| m.iter()
                            .filter_map(|(k, x)| x.as_u64().map(|s| (k.clone(), s)))
                            .collect())
                        .unwrap_or_default();
                    // rev 8: influence sketches ride alongside the scores;
                    // entries clamp into i32 (the committed sketch type).
                    let sketches = v["sketches"].as_object()
                        .map(|m| m.iter()
                            .filter_map(|(k, x)| x.as_array().map(|a| (
                                k.clone(),
                                a.iter().map(|e| e.as_i64().unwrap_or(0)
                                    .clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                                    .collect::<Vec<i32>>(),
                            )))
                            .collect())
                        .unwrap_or_default();
                    let _ = ev.send(FromBridge::Scores {
                        height: v["height"].as_u64().unwrap_or(0),
                        scores,
                        sketches,
                    }).await;
                }
                _ => {}
            }
        }
        #[allow(unreachable_code)]
        Ok::<(), std::io::Error>(())
    });

    let result = loop {
        tokio::select! {
            cmd = cmds.recv() => {
                let Some(cmd) = cmd else { break Ok(()) };
                let r = match cmd {
                    ToBridge::State { height, state, experts_per_layer } => {
                        let head = json!({"t": "state", "height": height,
                                          "n": state.len(),
                                          "experts_per_layer": experts_per_layer,
                                          "bin_next": true});
                        match write_frame(&mut wr, head.to_string().as_bytes()).await {
                            Ok(()) => write_frame(&mut wr,
                                &sestrian_core::int64_bytes(&state)).await,
                            e => e,
                        }
                    }
                    ToBridge::Train { height, seed, budget_s, min_nnz,
                                      max_nnz, quota_4dp, active_pages } => {
                        let m = json!({"t": "train", "height": height, "seed": seed,
                                       "budget_s": budget_s, "min_nnz": min_nnz,
                                       "max_nnz": max_nnz, "quota_4dp": quota_4dp,
                                       "active_pages": active_pages});
                        write_frame(&mut wr, m.to_string().as_bytes()).await
                    }
                    ToBridge::Advance { height, dim, sparse } => {
                        let m = json!({"t": "advance", "height": height,
                                       "dim": dim, "sparse": sparse});
                        write_frame(&mut wr, m.to_string().as_bytes()).await
                    }
                    ToBridge::Grow { height, new_dim, page_id, layer, expert, init } => {
                        let m = json!({"t": "grow", "height": height,
                                       "new_dim": new_dim,
                                       "page": {"page_id": page_id, "layer": layer,
                                                "expert": expert},
                                       "bin_next": true});
                        match write_frame(&mut wr, m.to_string().as_bytes()).await {
                            Ok(()) => write_frame(&mut wr,
                                &sestrian_core::int64_bytes(&init)).await,
                            e => e,
                        }
                    }
                    ToBridge::Generate { prompt, n } => {
                        let m = json!({"t": "generate", "prompt": prompt, "n": n});
                        write_frame(&mut wr, m.to_string().as_bytes()).await
                    }
                    ToBridge::Eval { height, seed, deltas } => {
                        let ds: Vec<Value> = deltas.iter()
                            .map(|(txid, sp)| json!({"txid": txid, "sparse": sp}))
                            .collect();
                        let m = json!({"t": "eval", "height": height,
                                       "seed": seed, "deltas": ds});
                        write_frame(&mut wr, m.to_string().as_bytes()).await
                    }
                };
                if let Err(e) = r {
                    break Err(e);
                }
            }
            res = &mut reader => {
                break match res {
                    Ok(inner) => inner,
                    Err(join) => Err(std::io::Error::other(join)),
                };
            }
        }
    };
    reader.abort();
    result
}

/// Run the bridge listener forever; one trainer at a time, reconnects welcome.
pub async fn run(
    port: u16,
    mut cmds: mpsc::Receiver<ToBridge>,
    events: mpsc::Sender<FromBridge>,
) {
    let listener = match TcpListener::bind(("127.0.0.1", port)).await {
        Ok(l) => l,
        Err(e) => {
            warn!("bridge listener failed: {e}");
            return;
        }
    };
    info!("trainer bridge listening on 127.0.0.1:{port}");
    loop {
        match listener.accept().await {
            Ok((sock, _)) => {
                if let Err(e) = serve_one(sock, &mut cmds, &events).await {
                    warn!("bridge connection ended: {e}");
                }
            }
            Err(e) => warn!("bridge accept error: {e}"),
        }
    }
}
