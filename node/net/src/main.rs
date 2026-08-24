//! sestrian-node — the production Rust node.
//!
//!   # a producing node with a PyTorch trainer attached:
//!   sestrian-node --data-dir ~/.sestrian/node --wallet ~/.sestrian/wallet.json \
//!       --port 7900 --api-port 8090 --bridge-port 7999 --produce \
//!       --peers /ip4/…/udp/7900/quic-v1 --data-contributor <addr>
//!   python -m client.miner_bridge --node-port 7999 --model small …
//!
//!   # a seed/relay node (always-on bootstrap; relays NAT'd peers):
//!   sestrian-node --data-dir /var/sestrian --key-seed <hex32> \
//!       --port 7900 --api-port 8090 --relay-server
//!
//! Genesis: --genesis-file <raw i64-LE .bin> (the ceremony artifact from
//! client/make_genesis.py), or fetched from peers against the published id.
//! The wallet key IS the miner identity; encrypted wallets are decrypted with
//! $SESTRIAN_WALLET_PASSPHRASE (argon2id + XSalsa20-Poly1305, the exact
//! pynacl construction).

mod api;
mod bridge;
mod node;
mod proto;
mod store;

use clap::Parser;
use libp2p::{Multiaddr, SwarmBuilder};
use sestrian_core as core;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Baked-in network parameters — Bitcoin's `chainparams` model.
///
/// Consensus-critical values must NOT be user-supplied. Bitcoin hardcodes its
/// genesis and consensus constants and lets you pick a *network* (`-testnet`,
/// `-regtest`); there is no way to typo yourself onto a different chain. We
/// previously took the genesis id, the genesis-ledger data-contributor, and the
/// bootstrap peer as free-form flags — and omitting or mistyping any of them
/// silently produced a node that could never validate a block. These now live
/// here, in the binary, and a conflicting override is a hard error.
///
/// The genesis WEIGHTS can't be a literal (they're ~650MB), but they are
/// deterministically derived from (model, seed) — so we bake the recipe and the
/// expected state_root, generate/verify locally, and get the same guarantee.
struct NetworkParams {
    name: &'static str,
    /// sha256 of the canonical genesis weight bytes; every node must match
    genesis_state_root: &'static str,
    /// seeds the founding corpus into the genesis LEDGER — consensus state
    data_contributor: &'static str,
    /// how to reproduce the genesis weights (client/make_genesis.py)
    genesis_model: &'static str,
    genesis_seed: u64,
    /// Published, hash-verifiable genesis archive — the fast path. Empty for a
    /// network that publishes none, in which case only reproduction is offered.
    genesis_url: &'static str,
    bootstrap: &'static str,
    /// Seconds between proposal attempts. Not consensus — but it sets the
    /// trainer's per-round budget and how often you compete for a block, so a
    /// node running a wildly different cadence than the network wastes its work:
    /// tiny useless rounds, and blocks that lose fork choice. Ships with the
    /// network so a joiner cannot get it wrong by accident (the old 10s default
    /// against this 180s network gave a 6-second training budget).
    block_interval: f64,
    /// PROTOCOL v1: the consensus ModelSpec — the page table is a pure function
    /// of these + on-chain growth events, so they are chainparams exactly like
    /// the genesis root. (n_layers, d_model, d_ff, n_experts_initial, e_max,
    /// backbone_params); client/gossip.py's preset of the same name must agree.
    spec: (u64, u64, u64, u64, u64, u64),
    /// Blocks per capacity-retarget window (§9.4a). Consensus.
    retarget_window: u64,
}

/// The network's GenesisParams — handed to every BlockTree/replay/validation.
///
/// LOCAL-ONLY overrides: on `--network local` the retarget constants may be
/// tightened via env vars so a growth event fires inside a bounded test run
/// (scripts/growth-proof.sh — verification-matrix item 9). These are consensus
/// parameters, so every node of the local chain must set the SAME values; they
/// are deliberately ignored on any named network.
fn genesis_params(net: &NetworkParams) -> core::model_state::GenesisParams {
    let (n_layers, d_model, d_ff, n_experts_initial, e_max, backbone_params) = net.spec;
    let spec = core::model_state::ModelSpec {
        n_layers, d_model, d_ff, n_experts_initial, e_max, backbone_params,
    };
    let mut gp = core::model_state::GenesisParams::new(spec);
    gp.retarget_window = net.retarget_window;
    if net.name == "local" {
        let env_u64 = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
        let env_i64 = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<i64>().ok());
        if let Some(v) = env_u64("SESTRIAN_LOCAL_RETARGET_WINDOW") { gp.retarget_window = v; }
        if let Some(v) = env_i64("SESTRIAN_LOCAL_TARGET_DELTAS") { gp.target_deltas = v; }
        if let Some(v) = env_i64("SESTRIAN_LOCAL_QUOTA_MAX_4DP") { gp.quota_max_4dp = v; }
        if let Some(v) = env_i64("SESTRIAN_LOCAL_K_SUSTAIN") { gp.k_sustain = v; }
        if let Some(v) = env_u64("SESTRIAN_LOCAL_ANNOUNCE_LEAD") { gp.announce_lead = v; }
        if let Some(v) = env_u64("SESTRIAN_LOCAL_DELTA_MAX_NNZ") { gp.delta_max_nnz = v; }
    }
    gp
}

// devnet-genesis-3 (protocol v2 — the delta envelope): ~107.4M-param MoE,
// state commitment. CEREMONY NOTE: genesis_state_root below is the PREVIEW
// value from the pre-release build — docs/genesis-ceremony.md requires it to be
// regenerated and cross-verified on BOTH founder machines (MPS + CUDA) against
// the final tagged code before the release is published; scripts/release-genesis.sh
// refuses to publish on any mismatch.
const DEVNET: NetworkParams = NetworkParams {
    name: "devnet",
    genesis_state_root: "91bdcc281c0dbbd7b3bea3d38003e4c61565bcaa5fd8e7bfca296e6a4994ddb1",
    data_contributor: "3432d48fd6878b4f2e7a1e40cc15e112c512fae7",
    genesis_model: "small-moe",
    genesis_seed: 20260824,
    genesis_url: "https://github.com/sestrian/sestrian/releases/download/devnet-genesis-3/genesis.bin.zst",
    // Names first, then the literal IPs they currently point at. The names are
    // what let an anchor move hosts — repoint DNS instead of cutting a release
    // and hoping everyone upgrades. The IPs stay as a floor so this build still
    // bootstraps before those records exist, and so a DNS outage cannot isolate
    // the network. Dials run in parallel; a name that does not resolve is just
    // one failed attempt among several.
    bootstrap: "/dns4/anchor1.sestrian.com/udp/9800/quic-v1,\
                /dns4/anchor2.sestrian.com/udp/9800/quic-v1,\
                /ip4/169.58.211.248/udp/9800/quic-v1,\
                /ip4/13.140.32.27/udp/9800/quic-v1",
    block_interval: 180.0,
    spec: (6, 512, 2048, 8, 16, 6_628_352),   // == client SMALL_MOE_CFG
    retarget_window: 16,
};

/// A private/local chain: nothing is baked, everything is explicit. This is the
/// escape hatch for `scripts/devnet.sh`, tests, and anyone standing up their own
/// network — the analogue of Bitcoin's `-regtest`.
const LOCAL: NetworkParams = NetworkParams {
    name: "local",
    genesis_state_root: "",
    data_contributor: "",
    genesis_model: "toy-moe",
    genesis_seed: 1337,
    genesis_url: "",
    bootstrap: "",
    block_interval: 10.0,
    spec: (2, 64, 256, 4, 8, 67_712),         // == client TOY_MOE_CFG
    retarget_window: 16,
};

/// Every way to obtain a network's genesis — printed on every genesis failure
/// so the remedy is never a search through the docs.
///
/// Download first, deliberately. Both paths end at the same bytes and the node
/// hashes them against the baked-in state root either way, so the download is
/// no less safe — it just does not cost a multi-gigabyte PyTorch install. When
/// the only option we offered was "reproduce it", that install was the price of
/// finding out whether your machine could join at all.
fn genesis_recipe(network: &str) -> String {
    let n = network_params(network);
    let expected = if n.genesis_state_root.is_empty() { "<your own network's id>" }
                   else { n.genesis_state_root };
    let mut s = String::new();
    if !n.genesis_url.is_empty() {
        s.push_str(&format!(
            "  Download it (verified against the genesis id — no PyTorch needed):\n    \
             npx sestrian genesis\n  \
             or by hand:\n    \
             curl -fL -o genesis.bin.zst {}\n    \
             zstd -d genesis.bin.zst -o genesis.bin\n\n  \
             Or reproduce it yourself — deterministic, so it needs no trust:\n",
            n.genesis_url));
    }
    s.push_str(&format!(
        "  uv run --with torch --with numpy --with pynacl \\\n    \
         python -m client.make_genesis --model {} --seed {} --out genesis.bin\n  \
         (either way the weights must hash to {})",
        n.genesis_model, n.genesis_seed, expected));
    s
}

fn network_params(name: &str) -> &'static NetworkParams {
    match name {
        "devnet" => &DEVNET,
        "local" => &LOCAL,
        other => panic!("unknown --network '{other}' (known: devnet, local)"),
    }
}

/// Fill unset flags from the selected network and REJECT conflicting overrides.
/// Silence is the enemy here: a mismatch means a chain you can never join, so it
/// must fail at startup with an explanation, not at block 1 with nothing.
fn apply_network(args: &mut Args) -> &'static NetworkParams {
    let net = network_params(&args.network);
    let adopt = |field: &mut String, baked: &'static str, what: &str| {
        if baked.is_empty() {
            return;
        }
        if field.is_empty() {
            *field = baked.to_string();
        } else if field != baked {
            panic!("--{what} does not match the '{}' network.\n  yours: {}\n  {}: {}\n\
                    These are consensus parameters — a mismatch is a chain you can \
                    never join. Omit the flag to use the network's value, or pass \
                    --network local to run your own chain.",
                   net.name, field, net.name, baked);
        }
    };
    adopt(&mut args.data_contributor, net.data_contributor, "data-contributor");
    adopt(&mut args.genesis_hash, net.genesis_state_root, "genesis-hash");
    if args.peers.is_empty() && !net.bootstrap.is_empty() {
        args.peers = net.bootstrap.to_string();       // extra peers may be added
    }
    if args.interval <= 0.0 {
        args.interval = net.block_interval;           // cadence follows the network
    }
    net
}

#[derive(Parser, Debug)]
struct Args {
    /// Which network to join: `devnet` (the live network — genesis id, bootstrap
    /// peer and genesis-ledger parameters are baked in) or `local` (your own
    /// chain; supply everything yourself). Consensus values come from here, so
    /// you cannot misconfigure yourself onto a chain that will never validate.
    #[arg(long, default_value = "devnet")]
    network: String,
    #[arg(long, default_value = "sestrian-data")]
    data_dir: String,
    #[arg(long, default_value = "")]
    wallet: String,          // wallet.json (identity); or:
    #[arg(long, default_value = "")]
    key_seed: String,        // DEPRECATED: raw hex seed on argv (ps-visible!)
    #[arg(long, default_value = "")]
    key_file: String,        // path to a 0600 file holding a 32-byte hex seed
    #[arg(long, default_value = "")]
    genesis_file: String,    // raw i64-LE genesis vector (ceremony artifact)
    #[arg(long, default_value = "")]
    genesis_hash: String,    // published genesis id; a fresh node fetches +
                             // verifies the genesis from a peer against this
    #[arg(long, default_value_t = 7900)]
    port: u16,
    #[arg(long, default_value_t = 8090)]
    api_port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    api_bind: String,        // interface for the HTTP API/dashboard
    #[arg(long, default_value = "0.0.0.0")]
    listen_bind: String,     // p2p listen interface; pin to one NIC (e.g. the
                             // LAN IP) on hosts with docker/k8s/libvirt bridges
                             // so libp2p stops advertising unreachable addrs
    #[arg(long, default_value_t = 7999)]
    bridge_port: u16,
    #[arg(long, default_value = "")]
    peers: String,
    #[arg(long, default_value_t = false)]
    produce: bool,
    /// Seconds between proposal attempts. 0 (the default) adopts the network's
    /// cadence — see NetworkParams::block_interval. Override only if you know why.
    #[arg(long, default_value_t = 0.0)]
    interval: f64,
    #[arg(long, default_value_t = 0.0)]
    seconds: f64,            // 0 = run forever
    #[arg(long, default_value = "")]
    data_contributor: String,
    #[arg(long, default_value = "")]
    data_refs: String,       // rev 5: comma-separated data_hashes of the staked
                             // corpora this miner trains on; named on every delta
                             // for provenance (empty deltas are rejected)
    /// PREFLIGHT: verify this machine can actually contribute — peer reachable,
    /// genesis id matches, identity/disk usable, and (with --interval) whether a
    /// training round can finish inside the block window. Prints a verdict and
    /// exits without touching the chain. Run this BEFORE mining for hours.
    #[arg(long, default_value_t = false)]
    check: bool,
    #[arg(long, default_value_t = false)]
    relay_server: bool,      // seeds: relay NAT'd peers (circuit relay v2)
    #[arg(long, default_value = "")]
    external_address: String, // advertise a known public multiaddr
    #[arg(long, default_value_t = 8)]
    prune_depth: u64,
    /// DA retention window in blocks: shard sets for bodies deeper than this
    /// are DELETED (pruned node). 0 = archive node, keep everything — run the
    /// public anchors this way so joiners can always fetch deep history.
    #[arg(long, default_value_t = 0)]
    da_retain_blocks: u64,
    /// shared round-clock origin (epoch seconds) — aligns round timing (and
    /// the sortition politeness ladder) across machines; 0 = process start
    #[arg(long, default_value_t = 0.0)]
    t0: f64,
}

/// Decrypt a pynacl-encrypted wallet: argon2id(MODERATE) -> XSalsa20-Poly1305.
fn decrypt_wallet(enc: &serde_json::Value, passphrase: &str) -> Option<[u8; 32]> {
    use argon2::{Algorithm, Argon2, Params, Version};
    use crypto_secretbox::aead::Aead;
    use crypto_secretbox::{KeyInit, XSalsa20Poly1305};
    let salt = hex::decode(enc["salt"].as_str()?).ok()?;
    let blob = hex::decode(enc["blob"].as_str()?).ok()?;
    // libsodium argon2id13 MODERATE: opslimit 3, memlimit 256 MiB
    let params = Params::new(256 * 1024, 3, 1, Some(32)).ok()?;
    let a2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    a2.hash_password_into(passphrase.as_bytes(), &salt, &mut key).ok()?;
    let (nonce, ct) = blob.split_at(24);
    let cipher = XSalsa20Poly1305::new((&key).into());
    let sk = cipher.decrypt(nonce.into(), ct).ok()?;
    sk.try_into().ok()
}

/// Encrypt a seed the way the Python client does, so a wallet this node writes
/// is readable by `client.wallet` and vice versa: argon2id(MODERATE) over a
/// random salt, then XSalsa20-Poly1305 with the nonce prefixed to the blob.
fn encrypt_wallet(seed: &[u8; 32], passphrase: &str) -> serde_json::Value {
    use argon2::{Algorithm, Argon2, Params, Version};
    use crypto_secretbox::aead::Aead;
    use crypto_secretbox::{KeyInit, XSalsa20Poly1305};
    use rand::RngCore;
    use zeroize::Zeroize;

    let mut salt = [0u8; 16];
    let mut nonce = [0u8; 24];
    rand::rngs::OsRng.fill_bytes(&mut salt);
    rand::rngs::OsRng.fill_bytes(&mut nonce);

    // Must match _decrypt_sk: libsodium argon2id13 MODERATE = ops 3, mem 256MiB.
    let params = Params::new(256 * 1024, 3, 1, Some(32)).expect("argon2 params");
    let a2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; 32];
    a2.hash_password_into(passphrase.as_bytes(), &salt, &mut key)
        .expect("argon2 kdf");

    let ct = XSalsa20Poly1305::new((&key).into())
        .encrypt((&nonce).into(), seed.as_slice())
        .expect("wallet encryption");
    key.zeroize();

    let mut blob = nonce.to_vec();
    blob.extend_from_slice(&ct);
    serde_json::json!({ "salt": hex::encode(salt), "blob": hex::encode(blob) })
}

/// Create a wallet at `path` and return its seed.
///
/// Onboarding must not require a second toolchain. A new operator running
/// `sestrian run` has no wallet, and the old behaviour — panic and tell them to
/// go install Python and run `client.wallet new` — is the difference between
/// joining and giving up. The format written here is byte-compatible with the
/// Python client's version 2 record.
fn create_wallet(path: &str, passphrase: Option<String>) -> [u8; 32] {
    use rand::RngCore;
    use std::io::Write;
    let mut seed = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut seed);

    let key = core::Key::from_seed(seed);
    let pub_hex = key.pub_hex();
    let mut rec = serde_json::json!({
        "version": 2,
        "pub": pub_hex,
        "address": core::token::address(&pub_hex),
    });
    match passphrase {
        Some(pw) if !pw.is_empty() => {
            rec["enc"] = encrypt_wallet(&seed, &pw);
        }
        _ => {
            warn!("SESTRIAN_WALLET_PASSPHRASE is not set — writing the key \
                   UNENCRYPTED (0600). Fine for devnet; set a passphrase before \
                   this identity holds anything you care about.");
            rec["sk"] = serde_json::Value::String(hex::encode(seed));
        }
    }

    let p = std::path::Path::new(path);
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("cannot create wallet directory");
        }
    }
    // create_new: refuse to overwrite. Clobbering a wallet destroys an identity
    // and every token balance behind it, so a race here must fail, never win.
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600); // owner-only from the instant it exists, not after
    }
    let mut f = opts.open(p).unwrap_or_else(|e| {
        panic!("cannot create wallet {path}: {e}");
    });
    let body = serde_json::to_vec_pretty(&rec).expect("serialize wallet");
    f.write_all(&body).expect("cannot write wallet");
    f.sync_all().expect("cannot flush wallet");

    warn!(wallet = path, address = %rec["address"].as_str().unwrap_or(""),
          "created a NEW wallet — BACK THIS FILE UP. It is your identity and \
           your balance, and nobody can recover it for you.");
    seed
}

/// Decode a 32-byte hex seed, wiping the hex text + decoded Vec afterwards so
/// transient key material doesn't linger in freed memory.
fn seed_from_hex(mut hexed: String) -> [u8; 32] {
    use zeroize::Zeroize;
    // Say what is wrong WITHOUT echoing key material. A bare "must be hex" panic
    // is useless when the real cause is upstream mangling — a YAML/CI layer
    // turning 64 digits into scientific notation, a stray newline, a truncated
    // paste. Report the shape, never the value.
    let trimmed = hexed.trim().to_string();
    let n = trimmed.len();
    let raw = hex::decode(&trimmed);
    hexed.zeroize();
    let mut raw = match raw {
        Ok(r) => r,
        Err(e) => {
            let mut t = trimmed;
            t.zeroize();
            panic!("key seed is not valid hex ({e}): got {n} characters, expected \
                    64 hex digits. If this came from CI or a config file, check \
                    it is QUOTED — a 64-digit unquoted value can be reinterpreted \
                    as a number.");
        }
    };
    let mut t = trimmed;
    t.zeroize();
    let out: [u8; 32] = raw.as_slice().try_into().unwrap_or_else(|_| {
        let len = raw.len();
        raw.zeroize();
        panic!("key seed must be 32 bytes ({len} decoded from {n} hex chars)")
    });
    raw.zeroize();
    out
}

/// Load the node identity WITHOUT ever taking key material from argv (which is
/// world-readable via ps/proc). Preferred sources, in order: a key file (0600),
/// the SESTRIAN_KEY_SEED env var, an (encrypted) wallet. --key-seed remains
/// only as a loud-deprecated fallback for local devnet.
fn load_identity(args: &Args) -> [u8; 32] {
    if !args.key_file.is_empty() {
        let hexed = std::fs::read_to_string(&args.key_file)
            .expect("--key-file unreadable");
        return seed_from_hex(hexed);
    }
    if let Ok(hexed) = std::env::var("SESTRIAN_KEY_SEED") {
        if !hexed.is_empty() {
            std::env::remove_var("SESTRIAN_KEY_SEED"); // don't leak to children
            return seed_from_hex(hexed);
        }
    }
    if !args.key_seed.is_empty() {
        warn!("--key-seed passes the private key on the command line, visible in \
               ps/proc to any local user; use --key-file or SESTRIAN_KEY_SEED");
        return seed_from_hex(args.key_seed.clone());
    }
    if !args.wallet.is_empty() {
        // First run: no wallet yet. Make one rather than sending the operator
        // away to another toolchain. Only when it is genuinely absent — an
        // unreadable existing file must still be a hard error, never quietly
        // replaced by a fresh identity.
        if !std::path::Path::new(&args.wallet).exists() {
            return create_wallet(
                &args.wallet,
                std::env::var("SESTRIAN_WALLET_PASSPHRASE").ok(),
            );
        }
        let raw = std::fs::read_to_string(&args.wallet).expect("wallet file unreadable");
        let w: serde_json::Value = serde_json::from_str(&raw).expect("wallet file corrupt");
        if let Some(sk) = w.get("sk").and_then(|s| s.as_str()) {
            return hex::decode(sk).unwrap().try_into().unwrap();
        }
        if let Some(enc) = w.get("enc") {
            let pw = std::env::var("SESTRIAN_WALLET_PASSPHRASE")
                .expect("encrypted wallet: set SESTRIAN_WALLET_PASSPHRASE");
            return decrypt_wallet(enc, &pw)
                .expect("wallet decryption failed (wrong passphrase?)");
        }
        panic!("wallet file has neither sk nor enc");
    }
    panic!("identity required. Easiest: --wallet <path> — if that file does not \
            exist yet the node creates one for you. Otherwise use --key-file or \
            the SESTRIAN_KEY_SEED environment variable.");
}

/// Resolve the genesis weights: local disk (durable) -> --genesis-file ->
/// --toy-dim -> FETCH from a peer, verified against the published --genesis-hash.
/// The genesis is public + self-verifying, so a fresh node bootstraps from a
/// single peer address plus the (tiny) published genesis id.
async fn resolve_genesis(args: &Args, store: &store::Store,
                         swarm: &mut libp2p::Swarm<node::Behaviour>,
                         gp: &core::model_state::GenesisParams) -> Vec<i64> {
    if let Some(g) = store.read_genesis() {
        ensure_genesis_dispersed(store, &g);
        return g; // durable once written
    }
    let g: Vec<i64> = if !args.genesis_file.is_empty() {
        let raw = std::fs::read(&args.genesis_file).expect("genesis file unreadable");
        raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    } else if !args.genesis_hash.is_empty() {
        info!(id = %args.genesis_hash, "no local genesis — fetching it from the network");
        // DA shards first: the only path that works at production scale, since
        // shards are fetched individually and never hit the sync response cap.
        // Falls back to the whole-genesis sync fetch (fine for toy/small chains).
        let fetched = match fetch_genesis_shards(swarm, store, &args.peers,
                                                 &args.genesis_hash, gp).await {
            Some(g) => Some(g),
            None => fetch_genesis(swarm, gp, &args.genesis_hash, &args.peers).await,
        };
        match fetched {
            Some(g) => g,
            // Peer-fetch only works for genesis vectors that fit the sync
            // response cap. The production model is ~650MB raw, far over it, so
            // this path fails by design there — don't leave the operator staring
            // at a 3-minute hang and a bare panic. The genesis is DETERMINISTIC,
            // so generating it locally is both faster and trustless.
            None => panic!("{}", [
                "could not obtain a genesis matching --genesis-hash from any peer",
                "(tried DA-shard reconstruction, then a whole-genesis fetch).",
                "",
                "No peer is serving genesis shards yet, or too few of them are.",
                "Generate the genesis locally instead — it is deterministic, so this",
                "is trustless, and usually faster than fetching it:",
                "",
                "  uv run --with torch --with numpy --with pynacl \\",
                "      python -m client.make_genesis --model <network model> --seed <network seed> \\",
                "      --out genesis.bin   # exact values: see the startup banner / docs/joining.md",
                "",
                "then re-run with --genesis-file genesis.bin (the printed",
                "genesis_state_root must equal the published genesis id).",
            ].join("\n")),
        }
    } else {
        panic!("{}", [
            "no genesis available — the node cannot validate a block without the",
            "weights. Get them either way:",
            "",
            &genesis_recipe(&args.network),
            "",
            "  …then re-run with --genesis-file genesis.bin",
        ].join("\n"));
    };
    // Whatever the source, the weights must hash to the network's baked-in
    // state_root — in protocol v1 that is the PAGE-MERKLE root over the
    // network ModelSpec's page table (what client/make_genesis.py prints).
    // This is the check that makes a wrong --genesis-file a startup error
    // instead of a node that quietly can't validate anything.
    if !args.genesis_hash.is_empty() {
        let model0 = core::model_state::ModelState::genesis(&gp.spec);
        if g.len() as u64 != model0.dim() {
            panic!("genesis length {} != the network ModelSpec's dimension {}\n\
                    (wrong model preset?)\n{}",
                   g.len(), model0.dim(), genesis_recipe(&args.network));
        }
        let got = core::model_state::page_state_root(&g, &model0);
        if got != args.genesis_hash
            && core::blocktree::genesis_block_hash(&g, gp) != args.genesis_hash {
            panic!("genesis does not match this network.\n  expected {}\n  got      {}\n\n\
                    Regenerate it, or pass --network local to run your own chain:\n{}",
                   args.genesis_hash, got, genesis_recipe(&args.network));
        }
        info!(params = g.len(), "genesis verified against the network's id");
    }
    store.write_genesis(&g).expect("cannot persist genesis");
    ensure_genesis_dispersed(store, &g);
    g
}

/// Erasure-code the genesis into DA shards once, so THIS node can serve it to
/// joiners. Every node that holds the genesis becomes a source, which is what
/// removes the single-host download and the local-regeneration requirement.
///
/// Costs, for the ~650MB production genesis: a few minutes of GF(256) work and
/// ~2GB of disk (48 shards x ~43MB, the same 3x redundancy delta bodies use).
/// One-time and idempotent — subsequent boots see the metadata and skip.
fn ensure_genesis_dispersed(store: &store::Store, g: &[i64]) {
    if store.shard_meta(store::Store::GENESIS_DA_KEY).is_some() {
        return; // already dispersed
    }
    let bytes = g.len() * 8;
    info!(params = g.len(), mb = bytes / (1 << 20),
          "dispersing the genesis into DA shards so peers can bootstrap from us \
           (one-time; this takes a while for a large model)");
    let t0 = std::time::Instant::now();
    match store.disperse_genesis(g) {
        Some(root) => info!(root = %&root[..16], secs = t0.elapsed().as_secs(),
                            "genesis dispersed"),
        None => warn!("could not disperse the genesis — this node will not be \
                       able to serve it to joiners"),
    }
}

/// Rebuild the genesis from erasure shards gathered across peers. Unlike the
/// whole-genesis sync fetch this has no size ceiling: each shard is its own
/// response, so a ~650MB genesis arrives as 16 x ~43MB pieces. Verifies the
/// result against the published id before adopting it, so a hostile peer can
/// at worst waste our time, never seed us a different chain.
async fn fetch_genesis_shards(swarm: &mut libp2p::Swarm<node::Behaviour>,
                              store: &store::Store, peers: &str,
                              expected_hash: &str,
                              gp: &core::model_state::GenesisParams)
                              -> Option<Vec<i64>> {
    use futures::StreamExt;
    use libp2p::{request_response, swarm::SwarmEvent};
    let key = store::Store::GENESIS_DA_KEY.to_string();
    let start = std::time::Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let connected: Vec<_> = swarm.connected_peers().copied().collect();
                if connected.is_empty() {
                    node::dial_peers(swarm, peers);
                }
                for p in connected {
                    swarm.behaviour_mut().shards.send_request(
                        &p, proto::ShardRequest { txids: vec![key.clone()] });
                }
                if start.elapsed().as_secs() > 180 {
                    return None;
                }
            }
            ev = swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(node::BehaviourEvent::Shards(
                    request_response::Event::Message {
                        message: request_response::Message::Response { response, .. },
                        peer, .. })) = ev
                {
                    let mut got = false;
                    for b in response.bodies.iter().filter(|b| b.txid == key) {
                        for (i, data) in &b.shards {
                            if let Some(bytes) = proto::unb64(data) {
                                store.put_shard(&key, *i, &bytes,
                                                b.k as usize, b.n as usize, b.orig_len);
                                got = true;
                            }
                        }
                    }
                    if got {
                        if let Some(g) = store.reconstruct_genesis() {
                            let model0 =
                                core::model_state::ModelState::genesis(&gp.spec);
                            if (g.len() as u64 == model0.dim()
                                && core::model_state::page_state_root(&g, &model0)
                                    == expected_hash)
                                || core::blocktree::genesis_block_hash(&g, gp)
                                    == expected_hash {
                                info!(params = g.len(),
                                      "genesis reconstructed from DA shards");
                                return Some(g);
                            }
                            warn!("genesis reconstructed from shards does NOT match \
                                   the published id — discarding");
                            return None;
                        }
                        // not enough shards yet — ask this peer again immediately
                        // (its cursor advances, so we get a different shard)
                        swarm.behaviour_mut().shards.send_request(
                            &peer, proto::ShardRequest { txids: vec![key.clone()] });
                    }
                }
            }
        }
    }
}

/// Fetch the genesis weights from a peer and verify they hash to the expected
/// genesis id before adopting them (so a malicious peer can't seed a wrong
/// genesis). Times out after a few minutes with no matching response.
/// PREFLIGHT (`--check`): answer "can this machine actually contribute?" before
/// the operator spends hours finding out it can't. Every check prints PASS/WARN/
/// FAIL with the concrete remedy — this exists because the failure modes here are
/// silent by nature (a too-slow trainer mines forever and earns nothing).
async fn preflight(args: &Args, key: &core::Key, store: &store::Store,
                   gp: &core::model_state::GenesisParams,
                   swarm: &mut libp2p::Swarm<node::Behaviour>)
                   -> Result<(), Box<dyn std::error::Error>> {
    use futures::StreamExt;
    let (mut fails, mut warns) = (0u32, 0u32);
    let pass = |m: String| println!("  \x1b[32mPASS\x1b[0m  {m}");
    println!("\nsestrian preflight — can this machine contribute?\n");

    // 1. identity + data dir (already opened above, so both are usable)
    pass(format!("identity loaded — miner {} / address {}",
                 &key.pub_hex()[..12], &core::token::address(&key.pub_hex())[..12]));
    pass(format!("data dir writable + exclusively locked ({})", args.data_dir));
    if store.read_genesis().is_some() {
        pass("existing chain on disk — will resume from it".into());
    }

    // 2. peer reachability: can we actually dial the bootstrap?
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut connected = 0usize;
    while std::time::Instant::now() < deadline {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                connected = swarm.connected_peers().count();
                if connected > 0 { break }
                node::dial_peers(swarm, &args.peers);
            }
            _ = swarm.select_next_some() => {
                connected = swarm.connected_peers().count();
                if connected > 0 { break }
            }
        }
    }
    if args.peers.is_empty() {
        println!("  \x1b[33mWARN\x1b[0m  no --peers given — this node will be alone \
                  (fine for a local devnet, not for joining)");
        warns += 1;
    } else if connected > 0 {
        pass(format!("bootstrap reachable — {connected} peer(s) connected"));
    } else {
        println!("  \x1b[31mFAIL\x1b[0m  cannot reach any peer in --peers within 30s.");
        println!("        Check the address/port and that outbound TCP is allowed.");
        fails += 1;
    }

    // 3. genesis agreement — the thing that silently forks you off the network
    // Verify whatever genesis we ALREADY have (on disk, or the given file)
    // against the network's id — that's the cheap, offline, authoritative check.
    // Only fall back to asking a peer when we have nothing locally.
    let local_genesis: Option<Vec<i64>> = store.read_genesis().or_else(|| {
        (!args.genesis_file.is_empty()).then(|| {
            let raw = std::fs::read(&args.genesis_file)
                .unwrap_or_else(|e| panic!("--genesis-file unreadable: {e}"));
            raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
        })
    });
    match (&local_genesis, args.genesis_hash.is_empty()) {
        (Some(g), false) => {
            let model0 = core::model_state::ModelState::genesis(&gp.spec);
            let root = if g.len() as u64 == model0.dim() {
                core::model_state::page_state_root(g, &model0)
            } else {
                format!("<wrong dimension: {} != {}>", g.len(), model0.dim())
            };
            if root == args.genesis_hash
                || (g.len() as u64 == model0.dim()
                    && core::blocktree::genesis_block_hash(g, gp) == args.genesis_hash) {
                pass(format!("genesis matches this network ({} params)", g.len()));
            } else {
                println!("  \x1b[31mFAIL\x1b[0m  your genesis does NOT match this network.");
                println!("        expected {}", args.genesis_hash);
                println!("        yours    {root}");
                println!("        You would be on a different chain. Regenerate it:");
                println!("{}", genesis_recipe(&args.network));
                fails += 1;
            }
        }
        (Some(g), true) => pass(format!("genesis present ({} params); this network \
                                         publishes no id to check it against", g.len())),
        (None, _) => {
            println!("  \x1b[31mFAIL\x1b[0m  no genesis weights — the node cannot \
                      validate anything without them.");
            println!("{}", genesis_recipe(&args.network));
            println!("  then re-run with --genesis-file genesis.bin");
            fails += 1;
        }
    }

    // 3b. --data-contributor is a GENESIS PARAMETER, not a preference: it seeds
    //     the founding corpus into the genesis ledger. Every node on a network
    //     must pass the identical value or its ledger diverges from block 1 and
    //     NOTHING will ever validate — the node sits at height 0 forever,
    //     receiving blocks and silently discarding them. (Observed exactly that
    //     while testing the join flow.)
    if !args.data_contributor.is_empty() {
        pass(format!("genesis-ledger parameter from the network — data-contributor {}",
                     &args.data_contributor[..12.min(args.data_contributor.len())]));
    } else if !args.peers.is_empty() {
        println!("  \x1b[33mWARN\x1b[0m  no data-contributor for this network — \
                  only correct if the chain was launched without one.");
        warns += 1;
    }

    // 4. the mining-viability check — the silent killer. A delta is includable
    //    only at base_height == head, so a round slower than the block interval
    //    is ALWAYS dropped. We can't time the GPU from here (that's the
    //    trainer's job, and it now auto-fits), but we can state the budget so
    //    the operator can compare it against their measured round time.
    if args.produce {
        let budget = args.interval * 0.6;
        pass(format!("producing with --interval {:.0}s → trainer budget ~{:.0}s \
                      per round (it auto-fits its steps to this)",
                     args.interval, budget));
        if args.interval < 60.0 {
            println!("  \x1b[33mWARN\x1b[0m  --interval {:.0}s is tight for a real \
                      network: multi-MB deltas may not propagate to the proposer \
                      in time, so your work would be orphaned. 120–180s is safer.",
                     args.interval);
            warns += 1;
        }
        if args.data_refs.is_empty() {
            println!("  \x1b[31mFAIL\x1b[0m  --produce without --data-refs: every \
                      delta you submit will be REJECTED (provenance is required).");
            println!("        Use --data-refs genesis, or the data_hash of a \
                      corpus you have staked.");
            fails += 1;
        } else {
            pass(format!("provenance set — deltas will name: {}", args.data_refs));
        }
    } else {
        pass("watch/serve mode (no --produce) — will sync and serve, not mine".into());
    }

    println!("\n{}\n", match (fails, warns) {
        (0, 0) => "\x1b[32mREADY\x1b[0m — this machine can contribute.".to_string(),
        (0, w) => format!("\x1b[33mREADY WITH {w} WARNING(S)\x1b[0m — see above."),
        (f, _) => format!("\x1b[31mNOT READY — {f} blocking problem(s)\x1b[0m."),
    });
    if fails > 0 { std::process::exit(1) }
    Ok(())
}

async fn fetch_genesis(swarm: &mut libp2p::Swarm<node::Behaviour>,
                       gp: &core::model_state::GenesisParams,
                       expected_hash: &str, peers: &str) -> Option<Vec<i64>> {
    use futures::StreamExt;
    use libp2p::{request_response, swarm::SwarmEvent};
    let start = std::time::Instant::now();
    let mut ticker = tokio::time::interval(Duration::from_secs(3));
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let connected: Vec<_> = swarm.connected_peers().copied().collect();
                if connected.is_empty() {
                    node::dial_peers(swarm, peers);
                }
                for p in connected {
                    swarm.behaviour_mut().sync.send_request(&p, proto::SyncRequest {
                        from_height: 0, count: 0, want_genesis: true });
                }
                if start.elapsed().as_secs() > 180 {
                    return None;
                }
            }
            ev = swarm.select_next_some() => {
                if let SwarmEvent::Behaviour(node::BehaviourEvent::Sync(
                    request_response::Event::Message {
                        message: request_response::Message::Response { response, .. }, .. })) = ev
                {
                    if let Some(w) = response.genesis {
                        // Accept either published form of the genesis id: the
                        // genesis block hash (header-format dependent) or the
                        // v1 page-Merkle state root (what make_genesis prints
                        // and the docs publish). Both pin the same weights.
                        let model0 = core::model_state::ModelState::genesis(&gp.spec);
                        if w.len() as u64 == model0.dim()
                            && (core::blocktree::genesis_block_hash(&w, gp) == expected_hash
                                || core::model_state::page_state_root(&w, &model0)
                                    == expected_hash) {
                            info!(dim = w.len(), "fetched + verified genesis from a peer");
                            return Some(w);
                        }
                        warn!("a peer served a genesis that doesn't match the published id");
                    }
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,libp2p=warn")))
        .init();
    let mut args = Args::parse();
    let net = apply_network(&mut args);
    let args = args;
    let gp = genesis_params(net);
    info!(network = net.name, "consensus parameters from the baked-in network");

    let seed = load_identity(&args);
    let consensus_key = core::Key::from_seed(seed);
    let p2p_key = libp2p::identity::Keypair::ed25519_from_bytes(seed)?;
    info!(miner = &consensus_key.pub_hex()[..12], "identity loaded");

    let store = store::Store::open(&args.data_dir)?;
    let dc = (!args.data_contributor.is_empty()).then(|| args.data_contributor.clone());

    // swarm with the full NAT stack: QUIC+TCP, Noise, relay client, AutoNAT,
    // DCUtR hole punching, optional relay server (seeds). Built BEFORE genesis
    // so a fresh node can fetch the genesis from a peer.
    let relay_server = args.relay_server;
    let mut swarm = SwarmBuilder::with_existing_identity(p2p_key)
        .with_tokio()
        .with_tcp(libp2p::tcp::Config::default(),
                  libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_quic_config(|mut cfg| {
            // libp2p-quic defaults (10s idle / 5s keepalive) are far too tight for
            // a NAT'd peer on a lossy internet path: a couple of dropped keepalives
            // and the connection idle-times-out, then redials — the ~30s connect/
            // drop cycle we saw. Give it a generous idle window with frequent
            // keepalives so NAT mappings stay warm and transient loss is survivable.
            cfg.max_idle_timeout = 120_000;                 // ms
            cfg.keep_alive_interval = Duration::from_secs(15);
            // Multi-MB sync responses over a lossy WAN killed connections with
            // quinn's INTERNAL_ERROR "too many gaps in stream buffer": with a
            // 10MB stream window, loss + reordering fragments the receiver's
            // reassembly buffer past quinn's gap limit whenever the event loop
            // pauses reading (block validation takes seconds). A 2MB stream
            // window bounds how fragmented a stream can ever get and still
            // sustains ~20MB/s at 100ms RTT — far above what catch-up needs.
            // Found live on the first transatlantic anchor sync.
            cfg.max_stream_data = 2_000_000;
            cfg.max_connection_data = 6_000_000;
            cfg
        })
        // Resolve /dns4 anchors. An anchor named in DNS can move to another
        // host without a binary release; a baked IP cannot, and every node
        // still running the old build would quietly never find the network.
        .with_dns()?
        .with_relay_client(libp2p::noise::Config::new, libp2p::yamux::Config::default)?
        .with_behaviour(|key, relay_client| {
            node::behaviour(key, relay_client, relay_server)
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();

    let topic = libp2p::gossipsub::IdentTopic::new("sestrian/v1");
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
    // preflight binds an EPHEMERAL port: it must be runnable while your real
    // node is already up, otherwise the check you most want to run is the one
    // you can't ("Address already in use").
    let listen_port = if args.check { 0 } else { args.port };
    swarm.listen_on(format!("/ip4/{}/udp/{}/quic-v1", args.listen_bind, listen_port).parse::<Multiaddr>()?)?;
    swarm.listen_on(format!("/ip4/{}/tcp/{}", args.listen_bind, listen_port).parse::<Multiaddr>()?)?;
    if !args.external_address.is_empty() {
        match args.external_address.parse::<Multiaddr>() {
            Ok(a) => swarm.add_external_address(a),
            Err(e) => warn!("bad --external-address: {e}"),
        }
    }
    node::dial_peers(&mut swarm, &args.peers);

    if args.check {
        return preflight(&args, &consensus_key, &store, &gp, &mut swarm).await;
    }

    // genesis: local disk -> --genesis-file -> --toy-dim -> FETCH from a peer,
    // verified against the published --genesis-hash. The genesis is public and
    // self-verifying, so a fresh node bootstraps from one peer + the id.
    resolve_genesis(&args, &store, &mut swarm, &gp).await;

    // replay any existing chain from disk (validated)
    let (tree, blocks_full, payloads) = store
        .replay(dc.clone(), args.prune_depth, &gp)
        .expect("chain replay failed");

    // Guarantee a current-format snapshot at the replayed head, so the NEXT
    // boot is a fast-boot even for an idle watcher that never advances to a
    // SNAPSHOT_EVERY height. Skips the write if disk already has one at head.
    if !matches!(store.read_snapshot(), Some((h, ..)) if h == tree.head) {
        let head = tree.head.clone();
        let height = tree.blocks[&head].height;
        store.write_snapshot(&head, height, tree.head_state(), tree.head_ledger(),
                             &tree.model[&head]);
        info!(height, "wrote boot snapshot for fast-boot");
    }

    // channels: api <-> node, bridge <-> node
    let (api_tx, api_rx) = mpsc::channel(64);
    let (bridge_cmd_tx, bridge_cmd_rx) = mpsc::channel::<bridge::ToBridge>(16);
    let (bridge_ev_tx, bridge_ev_rx) = mpsc::channel::<bridge::FromBridge>(16);
    let api_token = std::env::var("SESTRIAN_API_TOKEN").ok().filter(|t| !t.is_empty());
    tokio::spawn(api::run(args.api_bind.clone(), args.api_port, api_token, api_tx));
    tokio::spawn(bridge::run(args.bridge_port, bridge_cmd_rx, bridge_ev_tx));

    let n = node::Node {
        tree,
        store,
        key: consensus_key,
        blocks_full,
        da_pruned_to: 0,
        payloads,
        delta_pool: Default::default(),
        delta_scores: Default::default(),
        delta_sketches: Default::default(),
        omitted_deltas: Default::default(),
        account_pool: Default::default(),
        pending: Default::default(),
        pending_at: Default::default(),
        seen: Default::default(),
        seen_order: Default::default(),
        cfg: node::NodeConfig {
            produce: args.produce,
            interval: args.interval,
            seconds: args.seconds,
            peers: args.peers.clone(),
            data_refs: args.data_refs.split(',')
                .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            da_retain_blocks: args.da_retain_blocks,
        },
        topic,
        bridge_tx: bridge_cmd_tx,
        bridge_synced: false,
        train_inflight: false,
        train_deadline: 0.0,
        t0: if args.t0 > 0.0 { args.t0 } else {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?.as_secs_f64()
        },
        last_proposed_round: -1,
        last_trained_round: -1,
        sync_cursor: Default::default(),
        net: tokio::sync::mpsc::unbounded_channel().0, // replaced inside run()
        sync_walkback: Default::default(),
        peers_connected: 0,
        chat_pending: Vec::new(),
        chat_inflight: false,
        stale_deltas: 0,
        quota_rejects: 0,
        serve_shard_cursor: Default::default(),
        want_deltas: Default::default(),
    };
    node::run(n, swarm, api_rx, bridge_ev_rx).await;
    Ok(())
}

#[cfg(test)]
mod wallet_tests {
    use super::*;

    fn tmpdir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "sestrian-wallet-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The wallet a fresh operator gets must be internally consistent: the
    /// address in the file has to be the one the ledger will credit, and the
    /// pubkey has to be the one the seed actually signs with. If these drift,
    /// rewards go to an account nobody holds the key for.
    #[test]
    fn created_wallet_is_self_consistent() {
        let p = tmpdir().join("plain.json");
        let _ = std::fs::remove_file(&p);
        let seed = create_wallet(p.to_str().unwrap(), None);

        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        let pub_hex = rec["pub"].as_str().unwrap();

        assert_eq!(rec["version"], 2, "must be the v2 format the client reads");
        assert_eq!(core::Key::from_seed(seed).pub_hex(), pub_hex);
        assert_eq!(rec["address"].as_str().unwrap(), core::token::address(pub_hex));
        assert_eq!(rec["sk"].as_str().unwrap(), hex::encode(seed));
        assert!(rec.get("enc").is_none(), "unencrypted wallet must not carry enc");
    }

    /// Overwriting a wallet destroys an identity and every token behind it.
    #[test]
    fn refuses_to_overwrite_an_existing_wallet() {
        let p = tmpdir().join("nocobber.json");
        let _ = std::fs::remove_file(&p);
        create_wallet(p.to_str().unwrap(), None);
        let before = std::fs::read_to_string(&p).unwrap();

        let again = std::panic::catch_unwind(|| {
            create_wallet(p.to_str().unwrap(), None);
        });
        assert!(again.is_err(), "second create must fail, not replace the identity");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), before, "file was modified");
    }

    #[cfg(unix)]
    #[test]
    fn plaintext_wallet_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let p = tmpdir().join("perms.json");
        let _ = std::fs::remove_file(&p);
        create_wallet(p.to_str().unwrap(), None);
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "a key file readable by other users is a leak");
    }

    /// Encrypt must be the exact inverse of the decrypt path that was written
    /// to match pynacl — otherwise the node writes wallets only it can open.
    /// Slow by design: argon2id MODERATE is 256 MiB and crawls in a debug build.
    #[test]
    // Ignored in debug only because argon2id MODERATE (256 MiB, 3 passes) takes
    // ~16s unoptimised vs ~1s in release. CI runs it in release — see ci.yml.
    #[ignore = "slow unoptimised (~16s); CI runs it in release"]
    fn encrypted_wallet_round_trips() {
        let p = tmpdir().join("enc.json");
        let _ = std::fs::remove_file(&p);
        let seed = create_wallet(p.to_str().unwrap(), Some("a passphrase".into()));

        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert!(rec.get("sk").is_none(), "encrypted wallet must not also store the key");
        assert_eq!(decrypt_wallet(&rec["enc"], "a passphrase"), Some(seed));
        assert_eq!(decrypt_wallet(&rec["enc"], "wrong"), None);
    }
}
