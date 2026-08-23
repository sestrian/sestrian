#!/usr/bin/env python3
"""Stake a corpus on-chain: sign + submit a data_submit account tx.

The bytes are NOT uploaded (the API's 64MB cap can't take a multi-GB corpus —
see production-readiness "large-corpus DA ingestion"); the tx registers the
CONTENT HASH with a token stake, making the entry challengeable ("availability"
included) and namable by deltas' data_refs. Serve the bytes from an always-on
machine you control.

  python3 scripts/stake_corpus.py \
      --wallet ~/.sestrian/wallet.json --node http://127.0.0.1:8090 \
      --hash 85aa06fb... --size 18087897989 --stake-grains 100000000000

The registry id of the new entry is the TXID this prints — cite the DATA HASH
in --data-refs; validators resolve refs by hash.
"""
import argparse, json, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rig.crypto import Key                      # noqa: E402
from rig.token import DataSubmitTx, address     # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--node", default="http://127.0.0.1:8090")
    ap.add_argument("--hash", required=True, help="sha256 of the corpus bytes")
    ap.add_argument("--size", type=int, required=True, help="corpus size in bytes")
    ap.add_argument("--media", default="text")
    ap.add_argument("--stake-grains", type=int, required=True,
                    help="grains to escrow (1 SESTRIAN = 1e9 grains)")
    a = ap.parse_args()

    w = json.load(open(Path(a.wallet).expanduser()))
    key = Key.generate(bytes.fromhex(w["sk"]))
    if key.pub != w["pub"] or address(key.pub) != w["address"]:
        sys.exit("wallet file inconsistent: sk does not match pub/address")

    bal = json.load(urllib.request.urlopen(
        f"{a.node}/balance?addr={w['address']}", timeout=10))
    if bal["grains"] < a.stake_grains:
        sys.exit(f"balance {bal['grains']} < stake {a.stake_grains}")

    tx = DataSubmitTx(owner_pub=key.pub, data_hash=a.hash, size_bytes=a.size,
                      media_type=a.media, stake=a.stake_grains,
                      nonce=bal["nonce"]).signed(key)
    assert tx.verify()
    body = {"owner_pub": tx.owner_pub, "data_hash": tx.data_hash,
            "size_bytes": tx.size_bytes, "media_type": tx.media_type,
            "stake": tx.stake, "nonce": tx.nonce, "sig": tx.sig.hex()}
    req = urllib.request.Request(f"{a.node}/data/submit",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    print("txid (registry id):", tx.txid())
    print(urllib.request.urlopen(req, timeout=10).read().decode())
    print("pending — it enters the registry when the next block includes it; "
          "check /data/registry")


if __name__ == "__main__":
    main()
