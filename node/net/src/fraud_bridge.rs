//! JSON ↔ core::fraud bridge (Sharding Road P1). Fraud proofs travel as JSON
//! over gossip (core has no serde derives on Header/BackpropTx); this parses a
//! received proof into the core verifier's owned types and returns its verdict.

use sestrian_core::fraud::{verify, Body, PageFraudProof, TxRec};
use sestrian_core::{BackpropTx, Header};
use serde_json::Value;

fn header(v: &Value) -> Option<Header> {
    Some(Header {
        height: v["height"].as_u64()?,
        prev_hash: v["prev_hash"].as_str()?.into(),
        state_root: v["state_root"].as_str()?.into(),
        txset_root: v["txset_root"].as_str()?.into(),
        n_txs: v["n_txs"].as_u64()?,
        work: v["work"].as_u64()?,
        proposer: v["proposer"].as_str()?.into(),
        transfer_root: v["transfer_root"].as_str()?.into(),
        ledger_root: v["ledger_root"].as_str()?.into(),
        data_root: v["data_root"].as_str()?.into(),
        vrf_proof: v["vrf_proof"].as_str()?.into(),
        score_root: v["score_root"].as_str()?.into(),
        sketch_root: v["sketch_root"].as_str()?.into(),
        model_root: v["model_root"].as_str()?.into(),
        vrf_attempt: v["vrf_attempt"].as_u64()?,
        version: v["version"].as_u64()?,
    })
}

fn tx(v: &Value) -> Option<BackpropTx> {
    let pages = v["pages"].as_array()?.iter()
        .filter_map(|x| x.as_u64().map(|n| n as u32)).collect();
    let data_refs = v.get("data_refs").and_then(|d| d.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    Some(BackpropTx {
        miner: v["miner"].as_str()?.into(),
        base_height: v["base_height"].as_u64()?,
        delta_hash: v["delta_hash"].as_str()?.into(),
        da_pointer: v["da_pointer"].as_str()?.into(),
        bond: v["bond"].as_u64().unwrap_or(0),
        pages,
        data_refs,
        sig: hex::decode(v["sig_hex"].as_str()?).ok()?,
    })
}

/// Parse + verify. `Some((true, reason))` == the accused block is fraudulent;
/// `Some((false, reason))` == the proof does not convict; `None` == the proof
/// is unparseable (treat as no evidence).
pub fn fraud_verify(v: &Value) -> Option<(bool, String)> {
    let bodies = v["bodies"].as_object()?.iter().map(|(k, b)| {
        Some((k.clone(), Body {
            n: b["n"].as_u64()? as usize,
            idx: b["idx"].as_array()?.iter().filter_map(|x| x.as_u64()).collect(),
            val: b["val"].as_array()?.iter().filter_map(|x| x.as_i64()).collect(),
        }))
    }).collect::<Option<std::collections::HashMap<_, _>>>()?;
    let txs = v["txs"].as_array()?.iter()
        .map(|t| tx(t).map(|tx| TxRec { tx }))
        .collect::<Option<Vec<_>>>()?;
    let parent_path = v["parent_path"].as_array()?.iter().map(|e| {
        let a = e.as_array()?;
        Some((a[0].as_str()? == "L", a[1].as_str()?.to_string()))
    }).collect::<Option<Vec<_>>>()?;
    let proof = PageFraudProof {
        header: header(&v["header"])?,
        parent_header: header(&v["parent_header"])?,
        parent_model_json: v["parent_model_json"].as_str()?.into(),
        committed_leaves: v["committed_leaves"].as_array()?.iter()
            .filter_map(|x| x.as_str().map(String::from)).collect(),
        page_id: v["page_id"].as_u64()? as usize,
        txids: v["txids"].as_array()?.iter()
            .filter_map(|x| x.as_str().map(String::from)).collect(),
        txs,
        bodies,
        parent_page: v["parent_page"].as_array()?.iter()
            .filter_map(|x| x.as_i64()).collect(),
        parent_path,
    };
    Some(verify(&proof))
}
