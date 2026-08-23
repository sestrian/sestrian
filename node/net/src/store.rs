//! Disk persistence — a node must survive restarts with its chain intact.
//!
//! Layout under --data-dir:
//!   genesis.bin           raw i64-LE genesis weight vector
//!   blocks.jsonl          append-only StoredBlock per accepted block
//!   payloads/<txid>.json  compressed delta payloads (the DA bodies)
//!   snapshot.bin + snapshot.json   head-state checkpoint every SNAPSHOT_EVERY
//!
//! Boot = validated replay: headers/ledger replay from blocks.jsonl (our own
//! previously-validated data), state from the newest usable snapshot, then
//! full first-principles validation for every block after it. A corrupted or
//! missing snapshot silently degrades to full replay from genesis.

use crate::proto::{Payload, StoredBlock};
use sestrian_core::blocktree::BlockTree;
use sestrian_core::token::TokenLedger;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tracing::{info, warn};

pub const SNAPSHOT_EVERY: u64 = 25;

/// (tree, block-index for serving, in-memory payload cache for recent blocks)
type Rebuilt = (BlockTree, HashMap<String, StoredBlock>, HashMap<String, Payload>);

pub struct Store {
    dir: PathBuf,
    /// Held for the process lifetime: an advisory flock on the data dir. The
    /// kernel releases it automatically when the process exits (fd closed), so a
    /// crash never leaves a stale lock. Two processes on one --data-dir would
    /// interleave appends into blocks.jsonl and corrupt the chain.
    _lock: fs::File,
}

impl Store {
    pub fn open(dir: &str) -> std::io::Result<Store> {
        let dir = PathBuf::from(dir);
        fs::create_dir_all(dir.join("payloads"))?;
        let lock = Self::acquire_lock(&dir)?;
        Ok(Store { dir, _lock: lock })
    }

    /// Take an exclusive, non-blocking lock on `<dir>/.lock`. Two writers on one
    /// data dir interleave appends into blocks.jsonl and corrupt the chain, so
    /// this is a hard guarantee on every platform.
    ///
    /// Unix: advisory `flock`. Windows: opening with an empty share mode, which
    /// makes the OS itself refuse a second opener. Both are released by the OS
    /// when the process dies — the property that matters after a crash.
    #[cfg(unix)]
    fn acquire_lock(dir: &PathBuf) -> std::io::Result<fs::File> {
        use std::os::unix::io::AsRawFd;
        let path = dir.join(".lock");
        let file = fs::OpenOptions::new().create(true).write(true).truncate(false).open(&path)?;
        // SAFETY: valid fd for the lifetime of `file`; LOCK_NB never blocks.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("another sestrian-node already holds {}", path.display()),
            ));
        }
        Ok(file)
    }

    #[cfg(windows)]
    fn acquire_lock(dir: &PathBuf) -> std::io::Result<fs::File> {
        use std::os::windows::fs::OpenOptionsExt;
        let path = dir.join(".lock");
        // share_mode(0): no other process may open this file at all, so a second
        // node fails right here instead of racing us for the data dir.
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .share_mode(0)
            .open(&path)
            .map_err(|e| {
                // ERROR_SHARING_VIOLATION (32) is the one that means "someone
                // else has it open". Everything else — missing dir, read-only
                // volume, ACL — must keep its own message, or we would blame a
                // second node for a problem that has nothing to do with one.
                const ERROR_SHARING_VIOLATION: i32 = 32;
                if e.raw_os_error() == Some(ERROR_SHARING_VIOLATION) {
                    std::io::Error::new(
                        std::io::ErrorKind::WouldBlock,
                        format!("another sestrian-node already holds {}", path.display()),
                    )
                } else {
                    e
                }
            })
    }

    // ---- genesis ---------------------------------------------------------
    pub fn write_genesis(&self, w: &[i64]) -> std::io::Result<()> {
        let path = self.dir.join("genesis.bin");
        if path.exists() {
            return Ok(());
        }
        fs::write(path, sestrian_core::int64_bytes(w))
    }

    pub fn read_genesis(&self) -> Option<Vec<i64>> {
        let raw = fs::read(self.dir.join("genesis.bin")).ok()?;
        Some(raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }

    // ---- uploaded corpus files (DA custody for /upload submissions) ------
    pub fn save_upload(&self, hash: &str, bytes: &[u8]) -> std::io::Result<()> {
        let dir = self.dir.join("uploads");
        fs::create_dir_all(&dir)?;
        let path = dir.join(hash);
        if !path.exists() {
            fs::write(path, bytes)?;
        }
        Ok(())
    }

    // ---- payloads --------------------------------------------------------
    /// Persist a payload (idempotent). Written to a temp file + rename so a
    /// crash can't leave a half-written body that get_payload would accept, and
    /// fsync'd for durability. Returns whether it is now on disk — the caller
    /// (which drops the in-memory copy once a delta is mined) must not discard
    /// memory if this reports false.
    pub fn put_payload(&self, txid: &str, p: &Payload) -> bool {
        let path = self.dir.join("payloads").join(format!("{txid}.json"));
        if path.exists() {
            return true;
        }
        let tmp = self.dir.join("payloads").join(format!("{txid}.json.tmp"));
        let write = (|| -> std::io::Result<()> {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&serde_json::to_vec(p).unwrap())?;
            f.sync_all()?;
            fs::rename(&tmp, &path)
        })();
        if let Err(e) = write {
            warn!(%txid, "failed to persist payload: {e}");
            let _ = fs::remove_file(&tmp);
            return false;
        }
        // DA: erasure-code the body into shards so it stays recoverable even if
        // the monolithic file is later lost/pruned (best-effort; the monolithic
        // copy is the fast path, shards are the durability floor).
        let _ = self.disperse_payload(txid, p);
        true
    }

    pub fn get_payload(&self, txid: &str) -> Option<Payload> {
        // monolithic body first; if it's gone, reconstruct from erasure shards
        // (the DA layer) — so losing a body file never blocks replay/sync as
        // long as any K of the N shards survive.
        if let Ok(raw) = fs::read(self.dir.join("payloads").join(format!("{txid}.json"))) {
            if let Ok(p) = serde_json::from_slice(&raw) {
                return Some(p);
            }
        }
        self.reconstruct_payload(txid)
    }

    /// Delete a payload that will never be part of a block (a mempool delta
    /// evicted before inclusion) — reclaims disk from spammed/never-mined deltas.
    pub fn remove_payload(&self, txid: &str) {
        let _ = fs::remove_file(self.dir.join("payloads").join(format!("{txid}.json")));
        let _ = fs::remove_dir_all(self.dir.join("da").join(txid));
    }

    // ---- data availability: erasure-coded shards (§3.3) ------------------
    /// Reed-Solomon parameters: K data shards, N total. A body survives losing
    /// up to N-K shards, and any K reconstruct it.
    pub const DA_K: usize = 4;
    pub const DA_N: usize = 12;

    /// Reserved DA key for the genesis weight vector. Cannot collide with a txid
    /// (always 64 hex chars), so the existing shard-exchange protocol — which is
    /// keyed by string id — carries the genesis with no wire change.
    pub const GENESIS_DA_KEY: &'static str = "__genesis__";
    /// The genesis needs its own K/N. At the delta K=4 a ~650MB genesis shard
    /// would be ~163MB — far over the 96MB shard-response cap, so it could never
    /// be served. K=16 puts a shard at ~43MB (~57MB base64), which fits one per
    /// response. N=48 keeps the same 3x redundancy the delta bodies use.
    pub const GENESIS_DA_K: usize = 16;
    pub const GENESIS_DA_N: usize = 48;

    /// Erasure-code + Merkle-commit raw bytes into n shards under da/<key>/, so
    /// they are recoverable from any k. Returns the DA root (hex). Idempotent.
    ///
    /// Streams the shards to disk one at a time: the genesis is ~650MB and the
    /// all-at-once path would need several GB of RAM to hold the expanded blob.
    pub fn disperse_bytes(&self, key: &str, bytes: &[u8], k: usize, n: usize)
        -> Option<String>
    {
        let dir = self.dir.join("da").join(key);
        let meta_path = dir.join("meta.json");
        if let Ok(m) = fs::read(&meta_path) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&m) {
                match v["root"].as_str() {
                    Some(r) if !r.is_empty() => return Some(r.to_string()), // already dispersed
                    _ => {} // shards fetched from peers (root ""): re-derive below
                }
            }
        }
        fs::create_dir_all(&dir).ok()?;
        let mut write_err = None;
        let root = sestrian_core::da::disperse_streaming(bytes, k, n, |i, sh| {
            if let Err(e) = fs::write(dir.join(format!("{i}.shard")), sh) {
                write_err.get_or_insert(e);
            }
        });
        if let Some(e) = write_err {
            warn!(%key, "failed to write DA shards: {e}");
            return None;
        }
        let root = hex::encode(root);
        let meta = serde_json::json!({
            "root": root, "orig_len": bytes.len(), "k": k, "n": n,
        });
        let _ = fs::write(&meta_path, meta.to_string());
        Some(root)
    }

    /// Reconstruct raw bytes from whatever shards survive on disk (needs >= K).
    pub fn reconstruct_bytes(&self, key: &str) -> Option<Vec<u8>> {
        let dir = self.dir.join("da").join(key);
        let meta: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("meta.json")).ok()?).ok()?;
        let k = meta["k"].as_u64()? as usize;
        let orig_len = meta["orig_len"].as_u64()? as usize;
        let mut shards = std::collections::BTreeMap::new();
        for entry in fs::read_dir(&dir).ok()?.flatten() {
            let name = entry.file_name().into_string().ok()?;
            if let Some(i) = name.strip_suffix(".shard").and_then(|s| s.parse::<usize>().ok()) {
                if let Ok(data) = fs::read(entry.path()) {
                    shards.insert(i, data);
                }
            }
        }
        sestrian_core::da::reconstruct(&shards, k, orig_len)
    }

    /// Erasure-code + Merkle-commit a payload into N shards under da/<txid>/, so
    /// the body is recoverable from any K of them. Returns the DA root (hex) that
    /// commits the shard set. Idempotent.
    pub fn disperse_payload(&self, txid: &str, p: &Payload) -> Option<String> {
        let bytes = serde_json::to_vec(p).ok()?;
        self.disperse_bytes(txid, &bytes, Self::DA_K, Self::DA_N)
    }

    /// Reconstruct a payload from whatever shards survive on disk (needs >= K).
    /// None if too few shards remain (the body is unrecoverable locally).
    pub fn reconstruct_payload(&self, txid: &str) -> Option<Payload> {
        serde_json::from_slice(&self.reconstruct_bytes(txid)?).ok()
    }

    /// Disperse the GENESIS weights so this node can serve them to joiners as
    /// erasure shards. Shards sidestep the block-sync response cap entirely, so
    /// this is what lets a fresh node bootstrap the ~650MB genesis peer-to-peer
    /// instead of regenerating it locally (needs torch) or downloading it from a
    /// single host. Idempotent; safe to call on every boot.
    pub fn disperse_genesis(&self, g: &[i64]) -> Option<String> {
        self.disperse_bytes(Self::GENESIS_DA_KEY, &sestrian_core::int64_bytes(g),
                            Self::GENESIS_DA_K, Self::GENESIS_DA_N)
    }

    /// Rebuild the genesis weights from any K genesis shards on disk.
    pub fn reconstruct_genesis(&self) -> Option<Vec<i64>> {
        let raw = self.reconstruct_bytes(Self::GENESIS_DA_KEY)?;
        Some(raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// Shard indices we hold for a body, WITHOUT reading the shard bytes. The
    /// genesis shard set is ~2GB, so serving must never slurp it all to pick one.
    pub fn list_shard_indices(&self, key: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if let Ok(rd) = fs::read_dir(self.dir.join("da").join(key)) {
            for e in rd.flatten() {
                if let Ok(name) = e.file_name().into_string() {
                    if let Some(i) = name.strip_suffix(".shard")
                        .and_then(|s| s.parse::<u32>().ok()) {
                        out.push(i);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Read one shard by index.
    pub fn read_shard(&self, key: &str, i: u32) -> Option<Vec<u8>> {
        fs::read(self.dir.join("da").join(key).join(format!("{i}.shard"))).ok()
    }

    /// (k, n, orig_len) for a body's shard set, if we have its DA metadata.
    pub fn shard_meta(&self, txid: &str) -> Option<(usize, usize, u64)> {
        let m: serde_json::Value = serde_json::from_slice(
            &fs::read(self.dir.join("da").join(txid).join("meta.json")).ok()?).ok()?;
        Some((m["k"].as_u64()? as usize, m["n"].as_u64()? as usize, m["orig_len"].as_u64()?))
    }

    /// Every shard we currently hold for a body: (index, bytes).
    #[cfg(test)]   // production serving is budgeted+rotating; tests read all
    pub fn list_shards(&self, txid: &str) -> Vec<(u32, Vec<u8>)> {
        let mut out = Vec::new();
        if let Ok(rd) = fs::read_dir(self.dir.join("da").join(txid)) {
            for e in rd.flatten() {
                if let Ok(name) = e.file_name().into_string() {
                    if let Some(i) = name.strip_suffix(".shard").and_then(|s| s.parse::<u32>().ok()) {
                        if let Ok(d) = fs::read(e.path()) {
                            out.push((i, d));
                        }
                    }
                }
            }
        }
        out
    }

    /// Store a shard fetched from a peer (creating DA metadata if this is the
    /// first shard we've seen for this body).
    pub fn put_shard(&self, txid: &str, i: u32, data: &[u8], k: usize, n: usize, orig_len: u64) {
        let dir = self.dir.join("da").join(txid);
        let _ = fs::create_dir_all(&dir);
        let meta = dir.join("meta.json");
        if !meta.exists() {
            let _ = fs::write(&meta, serde_json::json!(
                {"k": k, "n": n, "orig_len": orig_len, "root": ""}).to_string());
        }
        let _ = fs::write(dir.join(format!("{i}.shard")), data);
    }

    /// Prune a body to a shard SUBSET: drop the monolithic payload and any shard
    /// not in `keep`. A node retains only its assigned shards of old bodies; the
    /// rest of the network holds the others, and any node can reconstruct by
    /// gathering K shards from peers.
    pub fn prune_body_to_shards(&self, txid: &str, keep: &[u32]) {
        let _ = fs::remove_file(self.dir.join("payloads").join(format!("{txid}.json")));
        if let Ok(rd) = fs::read_dir(self.dir.join("da").join(txid)) {
            for e in rd.flatten() {
                if let Ok(name) = e.file_name().into_string() {
                    if let Some(i) = name.strip_suffix(".shard").and_then(|s| s.parse::<u32>().ok()) {
                        if !keep.contains(&i) {
                            let _ = fs::remove_file(e.path());
                        }
                    }
                }
            }
        }
    }

    /// DA retention: drop EVERYTHING held for a body — payload and the whole
    /// shard dir. Used by pruned (non-archive) nodes once a block falls out of
    /// the retention window. Never called for the genesis key.
    pub fn delete_body_and_shards(&self, txid: &str) {
        let _ = fs::remove_file(self.dir.join("payloads").join(format!("{txid}.json")));
        let _ = fs::remove_dir_all(self.dir.join("da").join(txid));
    }

    // ---- block log -------------------------------------------------------
    /// Append + fsync a block. fsync makes the record durable across power loss;
    /// the caller MUST treat an Err as fatal (a dropped write silently truncates
    /// the chain on the next boot).
    pub fn append_block(&self, b: &StoredBlock) -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new().create(true).append(true)
            .open(self.dir.join("blocks.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(b).unwrap())?;
        f.sync_all()
    }

    /// Read the block log, tolerating a torn final record (a crash mid-append)
    /// by self-healing: truncate to the last good line. A corrupt record in the
    /// MIDDLE is real corruption — stop there and warn loudly rather than
    /// silently skipping it (skipping would orphan every block after it).
    pub fn read_blocks(&self) -> Vec<StoredBlock> {
        let Ok(raw) = fs::read_to_string(self.dir.join("blocks.jsonl")) else {
            return vec![];
        };
        let lines: Vec<&str> = raw.split('\n').collect();
        let mut out = Vec::new();
        let mut good_bytes = 0usize; // byte length of the fully-valid prefix
        for (i, line) in lines.iter().enumerate() {
            if line.is_empty() {
                if i + 1 == lines.len() {
                    break; // trailing "" after the final newline — clean EOF
                }
                good_bytes += 1; // a blank line's '\n'
                continue;
            }
            match serde_json::from_str::<StoredBlock>(line) {
                Ok(b) => {
                    out.push(b);
                    good_bytes += line.len() + 1; // + the '\n'
                }
                Err(e) => {
                    let more_after = lines[i + 1..].iter().any(|l| !l.is_empty());
                    if more_after {
                        warn!(line = i, err = %e,
                              "corrupt block record mid-log — replay stops here");
                    } else {
                        warn!(line = i, "torn final block record — truncating to last good line");
                        let _ = self.truncate_blocks(good_bytes);
                    }
                    break;
                }
            }
        }
        out
    }

    fn truncate_blocks(&self, len: usize) -> std::io::Result<()> {
        let f = fs::OpenOptions::new().write(true)
            .open(self.dir.join("blocks.jsonl"))?;
        f.set_len(len as u64)?;
        f.sync_all()
    }

    // ---- snapshots -------------------------------------------------------
    /// Checkpoint the full head state AND ledger, written atomically (temp +
    /// rename) so a crash mid-write can't leave a torn snapshot the fast path
    /// would trust. The state goes to a binary blob; hash/height/ledger to JSON.
    pub fn write_snapshot(&self, block_hash: &str, height: u64, state: &[i64],
                          ledger: &TokenLedger,
                          model: &sestrian_core::model_state::ModelState) {
        let bin_tmp = self.dir.join("snapshot.bin.tmp");
        if fs::write(&bin_tmp, sestrian_core::int64_bytes(state)).is_err() {
            return;
        }
        let _ = fs::rename(&bin_tmp, self.dir.join("snapshot.bin"));
        // format 2 (protocol v1): the ModelState rides as its CANONICAL JSON
        // string, so the fast-boot seed is byte-identical to what model_root
        // committed — a divergent fold can never sneak in through a snapshot.
        let meta = serde_json::json!({"format": 2, "hash": block_hash,
                                      "height": height,
                                      "ledger": ledger.to_value(),
                                      "model_state": model.canonical_json()});
        let json_tmp = self.dir.join("snapshot.json.tmp");
        if fs::write(&json_tmp, meta.to_string()).is_ok() {
            let _ = fs::rename(&json_tmp, self.dir.join("snapshot.json"));
        }
    }

    pub fn read_snapshot(&self)
        -> Option<(String, u64, Vec<i64>, TokenLedger,
                   sestrian_core::model_state::ModelState)> {
        let meta: serde_json::Value = serde_json::from_slice(
            &fs::read(self.dir.join("snapshot.json")).ok()?).ok()?;
        // reject pre-v1 snapshots (format < 2 / no model_state): seeding
        // without a ModelState would fork the fold; full replay instead.
        if meta["format"].as_u64() != Some(2) {
            warn!("snapshot is not format 2 (pre-v1) — ignoring, will full-replay");
            return None;
        }
        if !meta["ledger"].is_object() {
            warn!("snapshot has no ledger (old format) — ignoring, will full-replay");
            return None;
        }
        let model_json: serde_json::Value =
            serde_json::from_str(meta["model_state"].as_str()?).ok()?;
        let model = match sestrian_core::model_state::ModelState::from_json_value(
                &model_json) {
            Some(m) => m,
            None => {
                warn!("snapshot model_state malformed — ignoring, will full-replay");
                return None;
            }
        };
        let raw = fs::read(self.dir.join("snapshot.bin")).ok()?;
        let state = raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        // a malformed ledger => reject the whole snapshot => full validated replay
        let ledger = match TokenLedger::from_value(&meta["ledger"]) {
            Some(l) => l,
            None => {
                warn!("snapshot ledger is malformed — ignoring, will full-replay");
                return None;
            }
        };
        Some((meta["hash"].as_str()?.to_string(), meta["height"].as_u64()?, state,
              ledger, model))
    }

    /// Rebuild the tree + indices from disk. Tries FAST-BOOT from the newest
    /// snapshot (trust the checkpointed state/ledger, validate only the blocks
    /// after it); any problem falls back to full validated replay from genesis.
    pub fn replay(&self, data_contributor: Option<String>, prune_depth: u64,
                  params: &sestrian_core::model_state::GenesisParams)
        -> Option<Rebuilt>
    {
        let genesis = self.read_genesis()?;
        let blocks = self.read_blocks();
        if let Some((h, height, state, ledger, model)) = self.read_snapshot() {
            if let Some(r) = self.fast_replay(&genesis, &blocks, &data_contributor,
                                              prune_depth, &h, height, state,
                                              ledger, model, params) {
                return Some(r);
            }
            warn!("fast-boot unusable — falling back to full validated replay");
        }
        self.full_replay(genesis, &blocks, data_contributor, prune_depth, params)
    }

    /// Fast path: seed the tree with the snapshot's TRUSTED state+ledger at its
    /// block, cheaply index all headers (no payloads, no trimmed-mean), then run
    /// full validation forward from the snapshot only. Returns None (→ fallback)
    /// if the snapshot block isn't in the log or nothing validates past it.
    #[allow(clippy::too_many_arguments)]
    fn fast_replay(&self, genesis: &[i64], blocks: &[StoredBlock],
                   dc: &Option<String>, prune_depth: u64, snap_hash: &str,
                   snap_h: u64, snap_state: Vec<i64>, snap_ledger: TokenLedger,
                   snap_model: sestrian_core::model_state::ModelState,
                   params: &sestrian_core::model_state::GenesisParams)
        -> Option<Rebuilt>
    {
        let mut tree = BlockTree::new(genesis.to_vec(), dc.clone(), params.clone());
        tree.prune_depth = Some(prune_depth);

        // 1. headers + cum_work for every block up to the snapshot height, in
        //    height order so parents precede children (cheap — headers only).
        let mut sorted: Vec<&StoredBlock> = blocks.iter().collect();
        sorted.sort_by_key(|b| b.header.height);
        for sb in &sorted {
            if sb.header.height > snap_h {
                break;
            }
            let hdr = sb.header.to_core();
            if let Some(pw) = tree.cum_work.get(&hdr.prev_hash).copied() {
                let h = sb.hash();
                tree.blocks.insert(h.clone(), hdr.clone());
                tree.cum_work.insert(h, pw + hdr.work.max(1));
            }
        }
        if !tree.blocks.contains_key(snap_hash) {
            return None; // snapshot block not on disk — can't trust it
        }
        // 2. seed the checkpointed state + ledger + MODEL STATE at the
        //    snapshot block (v1: forward validation folds from this — the
        //    model_root commitment makes a divergent seed a loud failure)
        tree.state.insert(snap_hash.to_string(), snap_state);
        tree.ledger.insert(snap_hash.to_string(), snap_ledger);
        tree.model.insert(snap_hash.to_string(), snap_model);
        tree.head = snap_hash.to_string();

        // 3. index every stored block so we can still serve old ones; validate
        //    FORWARD only the blocks after the snapshot (add_block validates
        //    fully + runs fork choice, reconstructing their state/ledger).
        let mut index: HashMap<String, StoredBlock> =
            blocks.iter().map(|b| (b.hash(), b.clone())).collect();
        let mut cache = HashMap::new();
        let mut validated = 0u64;
        for sb in &sorted {
            if sb.header.height <= snap_h
                || !tree.state.contains_key(&sb.header.prev_hash) {
                continue;
            }
            let mut payloads = HashMap::new();
            for wt in &sb.txs {
                if let Some(t) = wt.to_core() {
                    if let Some(p) = self.get_payload(&t.txid()) {
                        payloads.insert(t.txid(), p);
                    }
                }
            }
            let Some(block) = sb.to_core(&payloads) else { continue };
            if tree.add_block(block).is_ok() {
                for (txid, p) in payloads { cache.insert(txid, p); }
                validated += 1;
            }
        }
        index.retain(|_, sb| sb.header.height <= tree.blocks[&tree.head].height);
        info!(from = snap_h, to = tree.blocks[&tree.head].height,
              validated, "FAST-BOOT from snapshot");
        Some((tree, index, cache))
    }

    fn full_replay(&self, genesis: Vec<i64>, blocks: &[StoredBlock],
                   dc: Option<String>, prune_depth: u64,
                   params: &sestrian_core::model_state::GenesisParams)
        -> Option<Rebuilt> {
        let mut tree = BlockTree::new(genesis, dc, params.clone());
        tree.prune_depth = Some(prune_depth);
        let mut index = HashMap::new();
        let mut cache = HashMap::new();
        for sb in blocks {
            let mut payloads = HashMap::new();
            for wt in &sb.txs {
                let Some(t) = wt.to_core() else { continue };
                if let Some(p) = self.get_payload(&t.txid()) {
                    payloads.insert(t.txid(), p);
                }
            }
            let Some(block) = sb.to_core(&payloads) else {
                warn!(hash = %sb.hash(), "block missing payloads — stopping replay");
                break;
            };
            match tree.add_block(block) {
                Ok(_) => {
                    for (txid, p) in payloads { cache.insert(txid, p); }
                    index.insert(sb.hash(), sb.clone());
                }
                Err(e) => {
                    warn!(hash = %sb.hash(), err = %e.0, "invalid block — stopping replay");
                    break;
                }
            }
        }
        info!(height = tree.blocks[&tree.head].height, blocks = index.len(),
              "full validated replay from genesis");
        Some((tree, index, cache))
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("sestrian-store-test-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&p);
        p
    }

    fn block_line(height: u64) -> String {
        // v1 wire form: every header field present — a line missing any of
        // them is a pre-fork artifact and read_blocks must NOT parse it.
        let hdr = format!(
            "{{\"height\":{height},\"prev_hash\":\"00\",\"state_root\":\"aa\",\
\"txset_root\":\"bb\",\"n_txs\":0,\"work\":1,\"proposer\":\"p\",\
\"transfer_root\":\"\",\"ledger_root\":\"\",\"data_root\":\"\",\
\"vrf_proof\":\"\",\"score_root\":\"\",\"sketch_root\":\"\",\
\"model_root\":\"cc\",\"vrf_attempt\":0,\"version\":1}}");
        format!("{{\"header\":{hdr},\"txs\":[],\"transfers\":[],\"data_txs\":[]}}")
    }

    #[test]
    fn pre_v1_block_lines_fail_loudly() {
        // the old (defaulted) wire format must not half-parse into a block
        let legacy = "{\"header\":{\"height\":1,\"prev_hash\":\"00\",\
\"state_root\":\"aa\",\"txset_root\":\"bb\",\"n_txs\":0,\"work\":1,\
\"proposer\":\"p\",\"transfer_root\":\"\",\"ledger_root\":\"\",\
\"data_root\":\"\"},\"txs\":[],\"transfers\":[],\"data_txs\":[]}";
        assert!(serde_json::from_str::<StoredBlock>(legacy).is_err(),
                "pre-v1 stored blocks must be rejected, not defaulted");
    }

    #[test]
    fn data_dir_lock_is_exclusive() {
        let dir = tmpdir("lock");
        let s1 = Store::open(dir.to_str().unwrap()).expect("first open ok");
        assert!(Store::open(dir.to_str().unwrap()).is_err(),
                "a second open of the same data-dir must be rejected");
        drop(s1); // releasing the lock lets a fresh process take it
        assert!(Store::open(dir.to_str().unwrap()).is_ok(),
                "open after release must succeed");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_blocks_recovers_from_torn_final_line() {
        let dir = tmpdir("torn");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        // two good records + a torn final line, exactly as a crash mid-append leaves
        let path = dir.join("blocks.jsonl");
        let content = format!("{}\n{}\n{}",
            block_line(1), block_line(2), r#"{"ZZZTORN":{"height":3,"#);
        fs::write(&path, &content).unwrap();
        let blocks = store.read_blocks();
        assert_eq!(blocks.len(), 2, "recovers the two good records");
        assert_eq!(blocks[0].header.height, 1);
        assert_eq!(blocks[1].header.height, 2);
        // the torn tail was self-healed off disk
        assert!(!fs::read_to_string(&path).unwrap().contains("ZZZTORN"),
                "torn line must be truncated from disk");
        assert_eq!(store.read_blocks().len(), 2, "re-read is clean");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn da_recovers_body_from_shards() {
        let dir = tmpdir("da");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        let p = Payload { n: 7, idx: "AAECAwQF".into(), val: "BgcICQoL".into() };
        let txid = "deadbeefcafe";
        assert!(store.put_payload(txid, &p)); // writes monolithic + N shards

        // lose the monolithic body AND N-K shards — still recoverable (K remain)
        let _ = fs::remove_file(dir.join("payloads").join(format!("{txid}.json")));
        let (n, k) = (Store::DA_N, Store::DA_K);
        for i in 0..(n - k) {
            let _ = fs::remove_file(dir.join("da").join(txid).join(format!("{i}.shard")));
        }
        let got = store.get_payload(txid).expect("K shards must reconstruct the body");
        assert_eq!((got.n, got.idx, got.val), (p.n, p.idx.clone(), p.val.clone()),
                   "reconstructed payload must equal the original");

        // drop one more shard (now < K) → the body is unrecoverable locally
        let _ = fs::remove_file(dir.join("da").join(txid).join(format!("{}.shard", n - k)));
        assert!(store.get_payload(txid).is_none(), "below K shards must be unrecoverable");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reconstructs_from_shards_split_across_holders() {
        // The multi-node DA guarantee: no single holder has K shards, but their
        // UNION does — which is exactly what the peer shard-exchange gathers.
        let dir_a = tmpdir("da-a");
        let a = Store::open(dir_a.to_str().unwrap()).unwrap();
        let p = Payload { n: 9, idx: "AAECAwQFBgcI".into(), val: "CQoLDA0ODxAR".into() };
        let txid = "splitbody";
        a.disperse_payload(txid, &p);
        let (k, n, orig) = a.shard_meta(txid).unwrap();
        let all: Vec<(u32, Vec<u8>)> = a.list_shards(txid);
        assert!(all.len() == n && k == Store::DA_K);

        // holder A keeps only shards 0..(K-1) — fewer than K, can't reconstruct
        let a_keep: Vec<u32> = (0..(k as u32 - 1)).collect();
        a.prune_body_to_shards(txid, &a_keep);
        assert!(a.reconstruct_payload(txid).is_none(), "< K shards alone can't reconstruct");

        // holder B receives A's kept shards + one it holds itself → union >= K
        let dir_b = tmpdir("da-b");
        let b = Store::open(dir_b.to_str().unwrap()).unwrap();
        for (i, data) in all.iter().take(k) {
            b.put_shard(txid, *i, data, k, n, orig);
        }
        let got = b.reconstruct_payload(txid).expect("union of K shards reconstructs");
        assert_eq!((got.n, got.idx, got.val), (p.n, p.idx.clone(), p.val.clone()));
        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn genesis_round_trips_through_da_shards() {
        // The join path for a fresh node: reconstruct the genesis from erasure
        // shards instead of regenerating it (needs torch) or downloading it whole
        // (exceeds the sync response cap).
        let dir = tmpdir("da-genesis");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        let g: Vec<i64> = (0..5000i64).map(|i| i.wrapping_mul(7_919) - 1_234).collect();
        let root = store.disperse_genesis(&g).expect("dispersal must succeed");
        assert_eq!(root.len(), 64, "root is a hex sha256");
        assert_eq!(store.reconstruct_genesis().as_ref(), Some(&g),
                   "all shards present must reconstruct the exact genesis");

        // idempotent: a second call returns the same root without re-encoding
        assert_eq!(store.disperse_genesis(&g).as_deref(), Some(root.as_str()));

        let (k, n, orig) = store.shard_meta(Store::GENESIS_DA_KEY).unwrap();
        assert_eq!((k, n, orig), (Store::GENESIS_DA_K, Store::GENESIS_DA_N,
                                  (g.len() * 8) as u64));
        assert_eq!(store.list_shard_indices(Store::GENESIS_DA_KEY).len(), n);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn genesis_reconstructs_from_exactly_k_shards_and_fails_below() {
        let dir = tmpdir("da-genesis-k");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        let g: Vec<i64> = (0..2048i64).map(|i| i * i - 3).collect();
        store.disperse_genesis(&g).unwrap();
        let (k, n, _) = store.shard_meta(Store::GENESIS_DA_KEY).unwrap();
        let da = dir.join("da").join(Store::GENESIS_DA_KEY);

        // drop N-K shards — exactly K remain, which must still reconstruct
        for i in 0..(n - k) {
            fs::remove_file(da.join(format!("{i}.shard"))).unwrap();
        }
        assert_eq!(store.list_shard_indices(Store::GENESIS_DA_KEY).len(), k);
        assert_eq!(store.reconstruct_genesis().as_ref(), Some(&g),
                   "exactly K shards must reconstruct");

        // one more gone → below K → unrecoverable, and it must fail cleanly
        fs::remove_file(da.join(format!("{}.shard", n - k))).unwrap();
        assert!(store.reconstruct_genesis().is_none(), "below K must not reconstruct");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn genesis_shards_gathered_from_a_peer_reconstruct() {
        // What the bootstrap actually does: a fresh node holds no genesis, pulls
        // K shards one at a time from a peer via put_shard, and rebuilds it.
        let dir_a = tmpdir("da-gen-a");
        let a = Store::open(dir_a.to_str().unwrap()).unwrap();
        let g: Vec<i64> = (0..3000i64).map(|i| 42 - i * 5).collect();
        a.disperse_genesis(&g).unwrap();
        let (k, n, orig) = a.shard_meta(Store::GENESIS_DA_KEY).unwrap();

        let dir_b = tmpdir("da-gen-b");
        let b = Store::open(dir_b.to_str().unwrap()).unwrap();
        assert!(b.reconstruct_genesis().is_none(), "fresh node has nothing");
        // pull K shards, checking it stays unrecoverable until the K'th arrives
        for (count, i) in a.list_shard_indices(Store::GENESIS_DA_KEY)
            .into_iter().take(k).enumerate()
        {
            let data = a.read_shard(Store::GENESIS_DA_KEY, i).unwrap();
            b.put_shard(Store::GENESIS_DA_KEY, i, &data, k, n, orig);
            if count + 1 < k {
                assert!(b.reconstruct_genesis().is_none(),
                        "only {} of {k} shards — must not reconstruct yet", count + 1);
            }
        }
        assert_eq!(b.reconstruct_genesis().as_ref(), Some(&g),
                   "K shards pulled from a peer rebuild the genesis exactly");
        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }

    /// Capacity check against a REAL genesis, for planning the one-time cost a
    /// node pays to become a genesis source. Ignored by default (needs a real
    /// genesis.bin and writes ~3x its size):
    ///   SESTRIAN_GENESIS_BIN=~/.sestrian/genesis.bin \
    ///     cargo test -p sestrian-node genesis_dispersal_cost -- --ignored --nocapture
    #[test]
    #[ignore]
    fn genesis_dispersal_cost() {
        let Ok(path) = std::env::var("SESTRIAN_GENESIS_BIN") else {
            eprintln!("set SESTRIAN_GENESIS_BIN to run this"); return;
        };
        let raw = fs::read(&path).expect("genesis.bin unreadable");
        let g: Vec<i64> = raw.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        let dir = tmpdir("da-genesis-cost");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        let t0 = std::time::Instant::now();
        let root = store.disperse_genesis(&g).expect("dispersal");
        let disperse_s = t0.elapsed().as_secs_f64();
        let shard_bytes = fs::metadata(dir.join("da").join(Store::GENESIS_DA_KEY)
            .join("0.shard")).unwrap().len();
        let t1 = std::time::Instant::now();
        let back = store.reconstruct_genesis().expect("reconstruct");
        let reconstruct_s = t1.elapsed().as_secs_f64();
        assert_eq!(back, g, "round trip must be exact");
        eprintln!("genesis {:.0}MB params={} -> k={} n={} shard={:.1}MB total={:.2}GB",
                  raw.len() as f64 / 1e6, g.len(), Store::GENESIS_DA_K,
                  Store::GENESIS_DA_N, shard_bytes as f64 / 1e6,
                  (shard_bytes as f64 * Store::GENESIS_DA_N as f64) / 1e9);
        eprintln!("disperse {disperse_s:.1}s   reconstruct {reconstruct_s:.1}s   root {}",
                  &root[..16]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_blocks_stops_at_midfile_corruption() {
        let dir = tmpdir("corrupt");
        let store = Store::open(dir.to_str().unwrap()).unwrap();
        // a corrupt record with a VALID record after it => real corruption:
        // stop at the corruption rather than silently skipping it.
        let content = format!("{}\n{}\n{}\n",
            block_line(1), "{not json}", block_line(3));
        fs::write(dir.join("blocks.jsonl"), &content).unwrap();
        let blocks = store.read_blocks();
        assert_eq!(blocks.len(), 1, "stops at the corrupt middle line");
        let _ = fs::remove_dir_all(&dir);
    }
}
