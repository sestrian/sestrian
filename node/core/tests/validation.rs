//! Negative validation tests: a validating node MUST reject malformed blocks.
//!
//! These pin the structural guards added for the production-readiness audit —
//! height linkage (task 90), n_txs / height-0 (task 95), and delta body length
//! (task 91) — plus the protocol-v1 rules: the version schedule, sortition
//! eligibility, page claims (frozen / zero-outside / quota), and the
//! model_root commitment. Each guard closes a concrete break: an unconstrained
//! height lets a miner mint the height-keyed reward forever; a
//! mismatched-length delta panics `trimmed_mean` on every validator (remote
//! chain-halt); an unenforced page claim makes a non-claimant's zero a vote
//! for zero. The positive path is covered by the golden vectors; this file
//! covers rejection.

use sestrian_core as core;
use sestrian_core::blocktree::{Block, BlockTree};
use sestrian_core::model_state::{GenesisParams, ModelSpec};
use std::collections::HashMap;

// toy v1 model: backbone (4 params) + one expert page (2*2 + 2 + 2*2 + 2 = 12)
const DIM: usize = 16;

fn test_params() -> GenesisParams {
    GenesisParams::new(ModelSpec {
        n_layers: 1,
        d_model: 2,
        d_ff: 2,
        n_experts_initial: 1,
        e_max: 2,
        backbone_params: 4,
    })
}

fn genesis_tree() -> BlockTree {
    BlockTree::new(vec![0i64; DIM], None, test_params())
}

/// A header whose roots are left blank but which carries a VALID proposer VRF
/// proof at attempt 0 (eligible under the cold-start rule: the genesis ledger
/// is empty) + matching work, so validation passes the lottery gate and
/// reaches the specific guard under test.
fn header(tree: &BlockTree, height: u64, n_txs: u64) -> core::Header {
    let key = core::Key::from_seed([9u8; 32]);
    let prev = tree.genesis_hash.clone();
    let proof = core::lottery::vrf_prove(&key, &prev, height, 0);
    core::Header {
        height,
        prev_hash: prev,
        state_root: String::new(),
        txset_root: String::new(),
        n_txs,
        work: core::lottery::attempt_work(&proof, 0),
        proposer: key.pub_hex(),
        transfer_root: String::new(),
        ledger_root: String::new(),
        data_root: String::new(),
        vrf_proof: hex::encode(&proof),
        score_root: String::new(),
        sketch_root: String::new(),
        model_root: String::new(),
        vrf_attempt: 0,
        version: 2,
    }
}

fn empty_block(header: core::Header) -> Block {
    Block { header, txs: vec![], bodies: HashMap::new(), sparse: HashMap::new(), transfers: vec![], data_txs: vec![],
            scores: Default::default(), sketches: Default::default() }
}

/// A correctly-signed page-claiming delta tx over `body` (hash matches).
fn page_tx(key: &core::Key, pages: Vec<u32>, body: &[i64]) -> core::BackpropTx {
    let dh = core::delta_hash(&core::int64_bytes(body));
    let mut tx = core::BackpropTx {
        miner: key.pub_hex(),
        base_height: 0,
        delta_hash: dh.clone(),
        da_pointer: format!("da://{dh}"),
        bond: 0,
        pages,
        data_refs: vec![],
        sig: vec![],
    };
    tx.sig = key.sign(&tx.signing_bytes());
    assert!(tx.verify());
    tx
}

/// A height-1 block carrying exactly `tx` with its DA body, blank roots (the
/// guards under test fire in the per-tx loop, before any root check).
fn tx_block(tree: &BlockTree, tx: core::BackpropTx, body: Vec<i64>) -> Block {
    let mut bodies = HashMap::new();
    bodies.insert(tx.da_pointer.clone(), body);
    Block {
        header: header(tree, 1, 1),
        txs: vec![tx],
        bodies,
        sparse: HashMap::new(),
        transfers: vec![],
        data_txs: vec![],
        scores: Default::default(),
        sketches: Default::default(),
    }
}

#[test]
fn rejects_wrong_height() {
    let mut tree = genesis_tree();
    // parent is genesis (height 0); a block claiming height 5 breaks the chain's
    // monotone height and must be rejected.
    let e = tree.add_block(empty_block(header(&tree, 5, 0))).unwrap_err();
    assert!(e.0.contains("height must be parent height + 1"), "got: {}", e.0);
}

#[test]
fn rejects_height_zero_nongenesis() {
    let mut tree = genesis_tree();
    // height 0 on a non-genesis block would underflow `h.height - 1`.
    let e = tree.add_block(empty_block(header(&tree, 0, 0))).unwrap_err();
    assert!(e.0.contains("height must be parent height + 1"), "got: {}", e.0);
}

#[test]
fn rejects_forged_work() {
    let mut tree = genesis_tree();
    let mut h = header(&tree, 1, 0); // valid VRF proof + correct vrf_work
    h.work = 999_999; // ...but claim an inflated fork-choice weight
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("VRF-derived weight"), "got: {}", e.0);
}

#[test]
fn rejects_invalid_vrf_proof() {
    let mut tree = genesis_tree();
    let mut h = header(&tree, 1, 0);
    h.vrf_proof = "00".repeat(64); // not a valid signature by the proposer
    let e = tree.add_block(empty_block(h)).unwrap_err();
    // v1: an unverifiable proof is simply never eligible
    assert!(e.0.contains("not eligible"), "got: {}", e.0);
}

#[test]
fn rejects_ntxs_mismatch() {
    let mut tree = genesis_tree();
    // header claims one tx but the block carries none.
    let e = tree.add_block(empty_block(header(&tree, 1, 1))).unwrap_err();
    assert!(e.0.contains("n_txs does not match"), "got: {}", e.0);
}

#[test]
fn rejects_wrong_length_delta_body() {
    let mut tree = genesis_tree();
    let key = core::Key::from_seed([7u8; 32]);
    // a correctly-signed, correctly-hashed delta whose body is the WRONG
    // dimension (3, not DIM=16) — this is exactly what would panic trimmed_mean.
    let body: Vec<i64> = vec![1, 2, 3];
    let tx = page_tx(&key, vec![0], &body);
    let e = tree.add_block(tx_block(&tree, tx, body)).unwrap_err();
    assert!(e.0.contains("delta body length"), "got: {}", e.0);
}

// --- protocol v1: version, eligibility, page claims, quota, model_root ------

#[test]
fn rejects_wrong_version() {
    let mut tree = genesis_tree();
    let mut h = header(&tree, 1, 0);
    h.version = 3; // not the scheduled version for this height
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("header version"), "got: {}", e.0);
}

#[test]
fn rejects_attempt_out_of_range() {
    let mut tree = genesis_tree();
    let mut h = header(&tree, 1, 0);
    h.vrf_attempt = core::lottery::ATTEMPT_MAX + 1;
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("vrf_attempt out of range"), "got: {}", e.0);
}

#[test]
fn rejects_ineligible_attempt() {
    // cold start (empty genesis ledger, supply 0): ONLY attempt 0 is eligible.
    // A valid proof at attempt 1 must be rejected — the widening ladder does
    // not apply before any stake exists.
    let mut tree = genesis_tree();
    let key = core::Key::from_seed([9u8; 32]);
    let prev = tree.genesis_hash.clone();
    let proof = core::lottery::vrf_prove(&key, &prev, 1, 1);
    let mut h = header(&tree, 1, 0);
    h.vrf_attempt = 1;
    h.vrf_proof = hex::encode(&proof);
    h.work = core::lottery::attempt_work(&proof, 1);
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("not eligible"), "got: {}", e.0);
}

#[test]
fn rejects_frozen_page_claim() {
    let mut tree = genesis_tree();
    // freeze the expert page in the parent ModelState; a delta claiming it is
    // invalid (frozen pages reject deltas but keep serving)
    let g = tree.genesis_hash.clone();
    tree.model.get_mut(&g).unwrap().pages[1].status = "F".into();
    let key = core::Key::from_seed([7u8; 32]);
    let mut body = vec![0i64; DIM];
    body[5] = 1; // inside the expert page span (4..16)
    let tx = page_tx(&key, vec![1], &body);
    let e = tree.add_block(tx_block(&tree, tx, body)).unwrap_err();
    assert!(e.0.contains("missing/frozen page"), "got: {}", e.0);
}

#[test]
fn rejects_missing_page_claim() {
    let mut tree = genesis_tree();
    let key = core::Key::from_seed([7u8; 32]);
    let mut body = vec![0i64; DIM];
    body[0] = 1;
    let tx = page_tx(&key, vec![0, 99], &body); // page 99 does not exist
    let e = tree.add_block(tx_block(&tree, tx, body)).unwrap_err();
    assert!(e.0.contains("missing/frozen page"), "got: {}", e.0);
}

#[test]
fn rejects_nonzero_outside_claims() {
    let mut tree = genesis_tree();
    let key = core::Key::from_seed([7u8; 32]);
    // claims only the expert page (span 4..16) but writes into the backbone —
    // a non-claimant's coordinate must be absence, not a vote
    let mut body = vec![0i64; DIM];
    body[5] = 1; // legitimately inside the claim
    body[0] = 7; // outside it
    let tx = page_tx(&key, vec![1], &body);
    let e = tree.add_block(tx_block(&tree, tx, body)).unwrap_err();
    assert!(e.0.contains("nonzero outside claimed pages"), "got: {}", e.0);
}

#[test]
fn rejects_below_quota() {
    let mut tree = genesis_tree();
    // raise the parent quota so the expert page (12 params) requires 12 nonzero
    // coordinates; a 1-nnz delta is below the work quota
    let g = tree.genesis_hash.clone();
    tree.model.get_mut(&g).unwrap().quota_4dp = 1_000_000;
    let key = core::Key::from_seed([7u8; 32]);
    let mut body = vec![0i64; DIM];
    body[5] = 1;
    let tx = page_tx(&key, vec![1], &body);
    let e = tree.add_block(tx_block(&tree, tx, body)).unwrap_err();
    assert!(e.0.contains("below work quota"), "got: {}", e.0);
}

#[test]
fn rejects_noncanonical_page_claims() {
    let mut tree = genesis_tree();
    let key = core::Key::from_seed([7u8; 32]);
    let mut body = vec![0i64; DIM];
    body[0] = 1;
    body[5] = 1;
    // unsorted claim set: the signed form is canonical, the carried form is not
    let tx = page_tx(&key, vec![1, 0], &body);
    let e = tree.add_block(tx_block(&tree, tx, body)).unwrap_err();
    assert!(e.0.contains("canonical"), "got: {}", e.0);
}

#[test]
fn rejects_wrong_model_root() {
    use sestrian_core::blocktree::{scores_root, sketch_root};
    use sestrian_core::model_state::page_state_root;
    let mut tree = genesis_tree();
    // an empty block whose state_root is CORRECT (no txs, no window boundary —
    // the post state equals the parent) but whose model_root is wrong: the fold
    // commitment must be validated independently of the weight commitment
    let mut h = header(&tree, 1, 0);
    let g = tree.genesis_hash.clone();
    h.state_root = page_state_root(tree.head_state(), &tree.model[&g]);
    h.txset_root = core::txset_root(&[]);
    h.score_root = scores_root(&Default::default());
    h.sketch_root = sketch_root(&Default::default());
    h.model_root = "11".repeat(32);
    let e = tree.add_block(empty_block(h)).unwrap_err();
    assert!(e.0.contains("model_root does not reproduce"), "got: {}", e.0);
}

// --- data-challenge market: quorum + disinterested jurors (task 93) ---------

use sestrian_core::token::{
    address, AccountTx, DataChallengeTx, DataSubmitTx, DataVoteTx, TokenLedger,
};
use std::collections::HashSet;

fn signed(mut tx: AccountTx, key: &core::Key) -> AccountTx {
    let sig = key.sign(&tx.signing_bytes());
    match &mut tx {
        AccountTx::Transfer(t) => t.sig = sig,
        AccountTx::DataSubmit(t) => t.sig = sig,
        AccountTx::DataChallenge(t) => t.sig = sig,
        AccountTx::DataVote(t) => t.sig = sig,
        AccountTx::InferenceReceipt(t) => t.sig = sig,
    }
    tx
}

/// Fund an owner, register a staked entry, fund a challenger, open a challenge.
/// Returns (ledger, owner, challenger, jurors, data_id, challenge_id).
fn open_challenge() -> (TokenLedger, core::Key, core::Key, Vec<core::Key>, String, String) {
    let owner = core::Key::from_seed([1u8; 32]);
    let challenger = core::Key::from_seed([2u8; 32]);
    let jurors: Vec<core::Key> = (10u8..13).map(|i| core::Key::from_seed([i; 32])).collect();
    let mut led = TokenLedger::new();
    // fund owner + challenger via block rewards
    led.apply_reward(1, &[owner.pub_hex()], &owner.pub_hex(), &[], &Default::default(), &Default::default());
    let sub = signed(AccountTx::DataSubmit(DataSubmitTx {
        owner_pub: owner.pub_hex(), data_hash: "aa".repeat(32), size_bytes: 8,
        media_type: "text".into(), stake: 1_000_000, nonce: 0, sig: vec![],
    }), &owner);
    assert!(led.apply_data_tx(&sub, 1, &HashSet::new()));
    let data_id = sub.txid();
    led.apply_reward(2, &[challenger.pub_hex()], &challenger.pub_hex(), &[], &Default::default(), &Default::default());
    let ch = signed(AccountTx::DataChallenge(DataChallengeTx {
        challenger_pub: challenger.pub_hex(), data_id: data_id.clone(), stake: 500_000,
        reason: "validity".into(), nonce: 0, sig: vec![],
    }), &challenger);
    assert!(led.apply_data_tx(&ch, 2, &HashSet::new()));
    let challenge_id = ch.txid();
    (led, owner, challenger, jurors, data_id, challenge_id)
}

#[test]
fn challenger_cannot_vote_on_own_challenge() {
    let (mut led, _owner, challenger, _jurors, _data_id, challenge_id) = open_challenge();
    let jset: HashSet<String> = [challenger.pub_hex()].into_iter().collect();
    let vote = signed(AccountTx::DataVote(DataVoteTx {
        voter_pub: challenger.pub_hex(), challenge_id, support: true, nonce: 1, sig: vec![],
    }), &challenger);
    // even though the challenger is a "recent proposer", they are an interested
    // party and must be rejected as a juror.
    assert!(!led.apply_data_tx(&vote, 3, &jset), "challenger self-vote must be rejected");
}

#[test]
fn owner_cannot_vote_on_challenge_of_own_entry() {
    let (mut led, owner, _challenger, _jurors, _data_id, challenge_id) = open_challenge();
    let jset: HashSet<String> = [owner.pub_hex()].into_iter().collect();
    let vote = signed(AccountTx::DataVote(DataVoteTx {
        voter_pub: owner.pub_hex(), challenge_id, support: false, nonce: 1, sig: vec![],
    }), &owner);
    assert!(!led.apply_data_tx(&vote, 3, &jset), "owner defending own entry must be rejected");
}

#[test]
fn challenge_below_quorum_is_rejected_and_refunds_owner() {
    use sestrian_core::token::CHALLENGE_QUORUM;
    let (mut led, owner, challenger, jurors, data_id, challenge_id) = open_challenge();
    let owner_addr = address(&owner.pub_hex());
    let bal_before = led.balance(&owner_addr);
    // only (QUORUM - 1) jurors uphold — below quorum
    let jset: HashSet<String> = jurors.iter().map(|k| k.pub_hex()).collect();
    for jk in jurors.iter().take(CHALLENGE_QUORUM - 1) {
        let vote = signed(AccountTx::DataVote(DataVoteTx {
            voter_pub: jk.pub_hex(), challenge_id: challenge_id.clone(), support: true,
            nonce: 0, sig: vec![],
        }), jk);
        assert!(led.apply_data_tx(&vote, 3, &jset));
    }
    let chal_addr = address(&challenger.pub_hex());
    let chal_before = led.balance(&chal_addr); // stake already escrowed out
    led.resolve_expired_challenges(2 + sestrian_core::token::CHALLENGE_WINDOW);
    // below quorum => NOT upheld: entry stays active, and the challenger's stake
    // is forfeited to the owner (lying/failed challenge costs).
    assert_eq!(led.registry[&data_id]["status"].as_str(), Some("active"),
               "sub-quorum challenge must not revoke the entry");
    assert_eq!(led.balance(&owner_addr), bal_before + 500_000,
               "owner must be refunded exactly the challenger's forfeited stake");
    // the challenger seized nothing back — its escrowed stake is gone to the owner
    assert_eq!(led.balance(&chal_addr), chal_before,
               "rejected challenger recovers nothing");
}

// --- adversarial ledger cases (task 126) ------------------------------------

use sestrian_core::token::TransferTx;

fn signed_transfer(from: &core::Key, to: &str, amount: u64, nonce: u64) -> TransferTx {
    let mut t = TransferTx {
        from_pub: from.pub_hex(), to_addr: to.into(), amount, nonce, sig: vec![],
    };
    t.sig = from.sign(&AccountTx::Transfer(t.clone()).signing_bytes());
    t
}

#[test]
fn rejects_replayed_and_overspending_transfers() {
    let sender = core::Key::from_seed([3u8; 32]);
    let recipient = address(&core::Key::from_seed([4u8; 32]).pub_hex());
    let mut led = TokenLedger::new();
    led.apply_reward(1, &[sender.pub_hex()], &sender.pub_hex(), &[], &Default::default(), &Default::default());
    let bal = led.balance(&address(&sender.pub_hex()));
    assert!(bal > 0);

    // a valid transfer applies once
    let t0 = signed_transfer(&sender, &recipient, bal / 4, 0);
    assert!(led.apply_transfer(&t0));
    // REPLAY: the same tx (nonce 0) can't apply again — nonce advanced
    assert!(!led.apply_transfer(&t0), "replayed transfer must be rejected");
    // OVERSPEND: amount beyond balance is rejected
    let huge = signed_transfer(&sender, &recipient, bal * 10, 1);
    assert!(!led.apply_transfer(&huge), "overspending transfer must be rejected");
    // WRONG NONCE: a future nonce (gap) can't apply yet
    let gap = signed_transfer(&sender, &recipient, 1, 5);
    assert!(!led.apply_transfer(&gap), "nonce gap must be rejected");
    // FORGED SIG: a tampered amount invalidates the signature
    let mut forged = signed_transfer(&sender, &recipient, 1, 1);
    forged.amount = 999_999;
    assert!(!led.apply_transfer(&forged), "tampered (bad-sig) transfer must be rejected");
}

#[test]
fn rejects_bad_data_lane_txs() {
    let owner = core::Key::from_seed([1u8; 32]);
    let mut led = TokenLedger::new();
    led.apply_reward(1, &[owner.pub_hex()], &owner.pub_hex(), &[], &Default::default(), &Default::default());

    // zero-stake submission is rejected
    let zero = signed(AccountTx::DataSubmit(DataSubmitTx {
        owner_pub: owner.pub_hex(), data_hash: "aa".repeat(32), size_bytes: 8,
        media_type: "text".into(), stake: 0, nonce: 0, sig: vec![],
    }), &owner);
    assert!(!led.apply_data_tx(&zero, 1, &HashSet::new()), "zero-stake submit rejected");

    // a valid submission, then a duplicate challenge on the same entry
    let sub = signed(AccountTx::DataSubmit(DataSubmitTx {
        owner_pub: owner.pub_hex(), data_hash: "bb".repeat(32), size_bytes: 8,
        media_type: "text".into(), stake: 1_000_000, nonce: 0, sig: vec![],
    }), &owner);
    assert!(led.apply_data_tx(&sub, 1, &HashSet::new()));
    let data_id = sub.txid();
    let challenger = core::Key::from_seed([2u8; 32]);
    led.apply_reward(2, &[challenger.pub_hex()], &challenger.pub_hex(), &[], &Default::default(), &Default::default());
    let ch = signed(AccountTx::DataChallenge(DataChallengeTx {
        challenger_pub: challenger.pub_hex(), data_id: data_id.clone(), stake: 100_000,
        reason: "validity".into(), nonce: 0, sig: vec![],
    }), &challenger);
    assert!(led.apply_data_tx(&ch, 2, &HashSet::new()));
    // a SECOND challenge against the same data_id (already open) is rejected
    let ch2 = signed(AccountTx::DataChallenge(DataChallengeTx {
        challenger_pub: challenger.pub_hex(), data_id, stake: 100_000,
        reason: "validity".into(), nonce: 1, sig: vec![],
    }), &challenger);
    assert!(!led.apply_data_tx(&ch2, 2, &HashSet::new()), "duplicate open challenge rejected");

    // a challenge against a non-existent entry is rejected
    let ghost = signed(AccountTx::DataChallenge(DataChallengeTx {
        challenger_pub: challenger.pub_hex(), data_id: "cc".repeat(32), stake: 100_000,
        reason: "validity".into(), nonce: 1, sig: vec![],
    }), &challenger);
    assert!(!led.apply_data_tx(&ghost, 2, &HashSet::new()), "challenge on missing entry rejected");
}

// --- snapshot (de)serialization hardening (task 94) -------------------------

#[test]
fn snapshot_roundtrips_and_rejects_malformed() {
    use serde_json::json;
    // a populated ledger (balances, nonces, a registry entry, an open challenge)
    let (led, _o, _c, _j, _d, _cid) = open_challenge();

    // a well-formed snapshot round-trips to the exact same root
    let good = led.to_value();
    let back = TokenLedger::from_value(&good).expect("valid snapshot must round-trip");
    assert_eq!(back.root(), led.root(), "round-trip must preserve the ledger root");

    // a non-integer balance is rejected (would corrupt supply / panic on math)
    let mut bad = led.to_value();
    let addr = bad["balances"].as_object().unwrap().keys().next().unwrap().clone();
    bad["balances"][addr.as_str()] = json!("not-a-number");
    assert!(TokenLedger::from_value(&bad).is_none(), "string balance must be rejected");

    // a registry entry missing a field is rejected (would panic apply_reward)
    let mut bad = led.to_value();
    let did = bad["registry"].as_object().unwrap().keys().next().unwrap().clone();
    bad["registry"][did.as_str()].as_object_mut().unwrap().remove("stake");
    assert!(TokenLedger::from_value(&bad).is_none(), "registry entry missing stake must be rejected");

    // a challenge whose vote list isn't an array of strings is rejected
    let mut bad = led.to_value();
    let cid = bad["challenges"].as_object().unwrap().keys().next().unwrap().clone();
    bad["challenges"][cid.as_str()]["votes_for"] = json!([1, 2, 3]);
    assert!(TokenLedger::from_value(&bad).is_none(), "non-string vote list must be rejected");

    // a missing top-level section is rejected
    let mut bad = led.to_value();
    bad.as_object_mut().unwrap().remove("challenges");
    assert!(TokenLedger::from_value(&bad).is_none(), "missing section must be rejected");
}

// --- signing-preimage framing is injective (task 96) ------------------------

#[test]
fn frame_resists_delimiter_injection() {
    // The classic collision the old '|'-joined signing strings allowed:
    // join(["a", "b|c"]) == "a|b|c" == join(["a|b", "c"]). Length-prefix framing
    // must keep these distinct, so a field's contents can never be re-parsed as
    // a different field split (which would give two txs the same txid).
    assert_ne!(core::frame(&[b"a", b"b|c"]), core::frame(&[b"a|b", b"c"]));
    // and empty-field boundaries are unambiguous too
    assert_ne!(core::frame(&[b"", b"ab"]), core::frame(&[b"a", b"b"]));
    // identical inputs still frame identically (determinism)
    assert_eq!(core::frame(&[b"x", b"yz"]), core::frame(&[b"x", b"yz"]));
}
