// Ad-hoc: reconstruct a REAL shard dir (path via env) — run with --ignored.
use std::collections::BTreeMap;

#[test]
#[ignore]
fn reconstruct_real_dir() {
    let dir = std::env::var("SHARD_DIR").expect("set SHARD_DIR");
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{dir}/meta.json")).unwrap()).unwrap();
    let k = meta["k"].as_u64().unwrap() as usize;
    let orig_len = meta["orig_len"].as_u64().unwrap() as usize;
    let mut shards = BTreeMap::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = e.file_name().into_string().unwrap();
        if let Some(i) = name.strip_suffix(".shard").and_then(|s| s.parse::<usize>().ok()) {
            shards.insert(i, std::fs::read(e.path()).unwrap());
        }
    }
    println!("k={k} orig_len={orig_len} shards={:?}",
             shards.keys().collect::<Vec<_>>());
    let t = std::time::Instant::now();
    let out = sestrian_core::da::reconstruct(&shards, k, orig_len);
    match &out {
        None => println!("RECONSTRUCT: None (FAILED) after {:?}", t.elapsed()),
        Some(b) => {
            println!("RECONSTRUCT: {} bytes in {:?}", b.len(), t.elapsed());
            let p: Result<serde_json::Value, _> = serde_json::from_slice(b);
            println!("JSON parse: {}", if p.is_ok() { "OK" } else { "FAILED" });
        }
    }
    assert!(out.is_some());
}
