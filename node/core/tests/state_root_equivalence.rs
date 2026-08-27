//! `state_root_with` (sparse) MUST equal materializing the post-state and
//! hashing it densely (`page_state_root`).
//!
//! The producer used to build a candidate's `state_root` by materializing a
//! full-length aggregate PLUS a full-length post-state: two ~915MB copies of the
//! model, at devnet scale, to express a delta that consensus caps at
//! `delta_max_nnz` coordinates. That peak — not any leak — is what the OOM
//! killer was reaping on the 7GB anchor. The producer now shares the validator's
//! incremental construction (reuse the cached leaf of every untouched page,
//! re-hash a touched page by streaming `canon[i] + agg[i]`).
//!
//! The two must agree BYTE FOR BYTE or a producer commits a root its own
//! validator rejects — the self-rejecting-blocks bug class. This file pins that
//! equivalence directly, over randomized aggregates, including the growth path
//! where pages are appended after aggregation and before the root.

use sestrian_core::blocktree::BlockTree;
use sestrian_core::model_state::{self, GenesisParams, ModelSpec, ModelState, Page};
use std::collections::BTreeMap;

fn spec() -> ModelSpec {
    // 2 layers x 3 experts: enough pages that "only touched pages rehash" is a
    // real distinction, and an odd leaf count so merkle's odd-promote is exercised.
    ModelSpec {
        n_layers: 2,
        d_model: 4,
        d_ff: 4,
        n_experts_initial: 3,
        e_max: 8,
        backbone_params: 32,
    }
}

fn params() -> GenesisParams {
    GenesisParams::new(spec())
}

fn dim(m: &ModelState) -> usize {
    m.pages.last().map(|p| p.end as usize).unwrap_or(0)
}

/// Deterministic PRNG — no dev-dependency, and a failure is reproducible from
/// its seed alone.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

/// The OLD dense construction, kept here verbatim as the reference oracle.
fn dense_root(canon: &[i64], agg: &BTreeMap<u32, i64>, init: &[Vec<i64>],
              post: &ModelState) -> String {
    let mut w = canon.to_vec();
    for (&i, &m) in agg {
        w[i as usize] = w[i as usize].wrapping_add(m);
    }
    for ip in init {
        w.extend(ip.iter().copied());
    }
    model_state::page_state_root(&w, post)
}

#[test]
fn sparse_state_root_equals_dense_no_growth() {
    let p = params();
    let m0 = ModelState::genesis(&spec());
    let n = dim(&m0);
    let mut rng = Lcg(0xC0FFEE);

    for trial in 0..200 {
        // a fresh genesis each trial so canon varies, not just the aggregate
        let canon: Vec<i64> = (0..n).map(|_| rng.next() as i64 % 10_000).collect();
        let tree = BlockTree::new(canon.clone(), None, p.clone());
        let spans: Vec<(u64, u64)> = m0.pages.iter().map(|q| (q.start, q.end)).collect();

        // sparse aggregate over a random subset — sometimes empty, sometimes
        // whole pages, so both the "untouched page reuses its leaf" and the
        // "every page touched" ends of the range get covered.
        let mut agg: BTreeMap<u32, i64> = BTreeMap::new();
        let k = (rng.next() % (n as u64 + 1)) as usize;
        for _ in 0..k {
            let i = (rng.next() % n as u64) as u32;
            let v = (rng.next() % 2_000) as i64 - 1_000;
            if v != 0 {
                agg.insert(i, v);
            }
        }

        let (sparse, _) = tree.state_root_with(&spans, &agg, &[]);
        let dense = dense_root(&canon, &agg, &[], &m0);
        assert_eq!(sparse, dense, "trial {trial}: sparse root diverged from dense");
    }
}

#[test]
fn sparse_state_root_equals_dense_with_growth() {
    let p = params();
    let m0 = ModelState::genesis(&spec());
    let n = dim(&m0);
    let expert_len = {
        // every expert page is the same size; take it from the last genesis page
        let last = m0.pages.last().unwrap();
        (last.end - last.start) as usize
    };
    let mut rng = Lcg(0xBEEF);

    for trial in 0..100 {
        let canon: Vec<i64> = (0..n).map(|_| rng.next() as i64 % 10_000).collect();
        let tree = BlockTree::new(canon.clone(), None, p.clone());
        let spans: Vec<(u64, u64)> = m0.pages.iter().map(|q| (q.start, q.end)).collect();

        let mut agg: BTreeMap<u32, i64> = BTreeMap::new();
        for _ in 0..(rng.next() % 64) {
            let i = (rng.next() % n as u64) as u32;
            let v = (rng.next() % 2_000) as i64 - 1_000;
            if v != 0 {
                agg.insert(i, v);
            }
        }

        // 1..=2 growth pages appended AFTER aggregation, BEFORE the root
        let n_grow = 1 + (rng.next() % 2) as usize;
        let mut init: Vec<Vec<i64>> = Vec::new();
        let mut post = m0.clone();
        for g in 0..n_grow {
            let page: Vec<i64> = (0..expert_len)
                .map(|_| rng.next() as i64 % 10_000).collect();
            let start = (n + g * expert_len) as u64;
            post.pages.push(Page {
                start,
                end: start + expert_len as u64,
                kind: "expert".into(),
                layer: 0,
                expert: (m0.pages.len() + g) as i64,
                status: "A".into(),
            });
            init.push(page);
        }

        let (sparse, _) = tree.state_root_with(&spans, &agg, &init);
        let dense = dense_root(&canon, &agg, &init, &post);
        assert_eq!(sparse, dense, "trial {trial}: sparse root diverged under growth");
    }
}

/// The equivalence must hold for the degenerate aggregate too: an empty one
/// leaves EVERY leaf cached, which is the path a block with no includable delta
/// would take.
#[test]
fn empty_aggregate_reproduces_the_parent_root() {
    let p = params();
    let m0 = ModelState::genesis(&spec());
    let n = dim(&m0);
    let canon: Vec<i64> = (0..n).map(|i| (i as i64) * 7 - 3).collect();
    let tree = BlockTree::new(canon.clone(), None, p);
    let spans: Vec<(u64, u64)> = m0.pages.iter().map(|q| (q.start, q.end)).collect();

    let (sparse, _) = tree.state_root_with(&spans, &BTreeMap::new(), &[]);
    assert_eq!(sparse, model_state::page_state_root(&canon, &m0));
}
