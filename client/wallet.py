"""Your wallet — an Ed25519 keypair on YOUR machine, never anywhere else.

The secret key lives in a mode-0600 file under ~/.sestrian/ (or --path),
ENCRYPTED with a passphrase (argon2id key derivation + libsodium SecretBox) —
and is never transmitted, never logged, and must NEVER enter a git repository.
On creation you also get a BIP39 24-word mnemonic: write it down; it alone can
restore the wallet (`restore`). Addresses display as checksummed bech32
(`pal1…`) so a typo cannot burn funds; the chain's internal form stays hex.

  python -m client.wallet new                  # create (passphrase + mnemonic)
  python -m client.wallet restore              # rebuild from the 24 words
  python -m client.wallet show                 # address + pubkey (never the secret)
  python -m client.wallet balance --node http://localhost:8090
  python -m client.wallet send --to pal1… --amount 1.5 --node http://localhost:8090

Extra deps for the hardened features: `pip install mnemonic bech32`
(both are the reference implementations of their standards).

For the REAL genesis ceremony: generate the founding wallet fresh, offline, on a
machine you trust — the mnemonic IS the wallet.
"""

import argparse
import getpass
import json
import os
import stat
import sys
import urllib.request

from rig.crypto import Key
from rig.token import GRAIN, TransferTx, address

DEFAULT_DIR = os.path.expanduser("~/.sestrian")
DEFAULT_PATH = os.path.join(DEFAULT_DIR, "wallet.json")
HRP = "pal"                                       # bech32 human-readable prefix


# ---- checksummed display addresses (bech32; internal hex is consensus) ------
def to_display(addr_hex: str) -> str:
    try:
        import bech32
        return bech32.bech32_encode(
            HRP, bech32.convertbits(bytes.fromhex(addr_hex), 8, 5))
    except ImportError:
        return addr_hex                            # graceful: hex still works

def parse_addr(s: str) -> str:
    """Accept pal1… (checksum-verified) or raw 40-hex; return internal hex."""
    if s.startswith(HRP + "1"):
        import bech32
        hrp, data = bech32.bech32_decode(s)
        if hrp != HRP or data is None:
            raise SystemExit(f"bad address (checksum failed): {s}")
        return bytes(bech32.convertbits(data, 5, 8, False)).hex()
    if len(s) == 40 and all(c in "0123456789abcdef" for c in s.lower()):
        return s.lower()
    raise SystemExit(f"not a valid address: {s}")


# ---- encrypted wallet file --------------------------------------------------
def _encrypt_sk(sk: bytes, passphrase: str) -> dict:
    from nacl import pwhash, secret, utils
    salt = utils.random(pwhash.argon2id.SALTBYTES)
    key = pwhash.argon2id.kdf(secret.SecretBox.KEY_SIZE, passphrase.encode(), salt,
                              opslimit=pwhash.argon2id.OPSLIMIT_MODERATE,
                              memlimit=pwhash.argon2id.MEMLIMIT_MODERATE)
    blob = secret.SecretBox(key).encrypt(sk)       # nonce included in blob
    return {"kdf": "argon2id", "salt": salt.hex(), "blob": bytes(blob).hex()}

def _decrypt_sk(enc: dict, passphrase: str) -> bytes:
    from nacl import pwhash, secret
    key = pwhash.argon2id.kdf(secret.SecretBox.KEY_SIZE, passphrase.encode(),
                              bytes.fromhex(enc["salt"]),
                              opslimit=pwhash.argon2id.OPSLIMIT_MODERATE,
                              memlimit=pwhash.argon2id.MEMLIMIT_MODERATE)
    return secret.SecretBox(key).decrypt(bytes.fromhex(enc["blob"]))


def _write_wallet(path: str, key: Key, passphrase: str) -> dict:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"version": 2, "pub": key.pub, "address": address(key.pub)}
    if passphrase:
        rec["enc"] = _encrypt_sk(key.sk, passphrase)
    else:
        print("⚠ empty passphrase — storing the key UNENCRYPTED (testnet only)")
        rec["sk"] = key.sk.hex()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=1)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)   # 0600 — owner only
    return rec


def _mnemonic_for(sk: bytes) -> str | None:
    try:
        from mnemonic import Mnemonic
        return Mnemonic("english").to_mnemonic(sk)   # 32 bytes -> 24 words
    except ImportError:
        return None


def create(path: str, passphrase_env: str | None = None) -> dict:
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing wallet: {path}")
    key = Key.generate()                          # 32 random bytes from os.urandom
    if passphrase_env is not None:
        # non-interactive path (installer, CI, provisioning): take the passphrase
        # from an env var so nothing lands in argv/ps. Unset or empty = plaintext.
        pw = os.environ.get(passphrase_env, "")
    elif not sys.stdin.isatty():
        # getpass would raise a bare EOFError here — useless to an operator
        raise SystemExit(
            "wallet new needs a terminal to prompt for a passphrase.\n"
            "Non-interactive? Use --passphrase-env VAR (unset/empty = "
            "unencrypted), e.g.:\n"
            "  SESTRIAN_WALLET_PASSPHRASE=... python -m client.wallet new "
            "--passphrase-env SESTRIAN_WALLET_PASSPHRASE")
    else:
        pw = getpass.getpass("passphrase for the wallet file (empty = unencrypted): ")
        if pw and pw != getpass.getpass("repeat passphrase: "):
            raise SystemExit("passphrases do not match")
    rec = _write_wallet(path, key, pw)
    words = _mnemonic_for(key.sk)
    if words:
        print("\n=== RECOVERY MNEMONIC — write these 24 words down, in order ===")
        print(words)
        print("=== anyone with these words owns the wallet; store them offline ===\n")
    else:
        print("(`pip install mnemonic` to get a recovery phrase next time)")
    return rec


def restore(path: str) -> dict:
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing wallet: {path}")
    try:
        from mnemonic import Mnemonic
    except ImportError:
        raise SystemExit("restore needs the reference BIP39 lib: pip install mnemonic")
    words = input("enter your 24-word mnemonic: ").strip()
    m = Mnemonic("english")
    if not m.check(words):
        raise SystemExit("mnemonic checksum failed — check the words and order")
    sk = bytes(m.to_entropy(words))
    key = Key.generate(sk)
    pw = getpass.getpass("new passphrase for the wallet file (empty = unencrypted): ")
    rec = _write_wallet(path, key, pw)
    print(f"restored: {to_display(rec['address'])}")
    return rec


def load(path: str) -> tuple[Key, dict]:
    with open(path) as f:
        rec = json.load(f)
    if "enc" in rec:
        pw = getpass.getpass("wallet passphrase: ")
        sk = _decrypt_sk(rec["enc"], pw)
    else:
        sk = bytes.fromhex(rec["sk"])              # legacy/unencrypted
    key = Key.generate(sk)
    assert key.pub == rec["pub"], "wallet file corrupt (pub mismatch)"
    return key, rec


def _get(node: str, route: str):
    with urllib.request.urlopen(f"{node}{route}", timeout=10) as r:
        return json.loads(r.read())


def _post(node: str, route: str, payload: dict):
    req = urllib.request.Request(
        f"{node}{route}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _submit(node: str, addr: str, route: str, build, tries: int = 3,
            wait_s: int = 240):
    """Submit a nonce-ordered tx and CONFIRM it landed, retrying on a nonce race.

    Every account tx is ordered by the sender's nonce, so two machines sharing a
    wallet (or a resubmit after a chain reset) will pick the same nonce and one
    of them is silently discarded — the CLI would print 'in mempool' and the tx
    would never apply. Poll until the nonce actually advances; if it doesn't,
    rebuild against the current nonce and resend.

    `build(nonce) -> (tx, payload)`. Returns the node's reply once confirmed.
    """
    import time
    for attempt in range(1, tries + 1):
        info = _get(node, f"/balance?addr={addr}")
        nonce = info.get("nonce", 0)
        tx, payload = build(nonce)
        out = _post(node, route, payload)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(5)
            if _get(node, f"/balance?addr={addr}").get("nonce", 0) > nonce:
                return out                       # nonce advanced ⇒ ours applied
        if attempt < tries:
            print(f"  not confirmed in {wait_s}s (nonce still {nonce}) — likely a "
                  f"nonce race; rebuilding and resending [{attempt}/{tries}]")
    raise SystemExit(
        f"tx never confirmed after {tries} attempts. If another machine shares "
        f"this wallet, submit from only one at a time.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "restore", "show", "balance", "send",
                                    "submit-data", "challenge", "vote", "registry"])
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--node", default="http://localhost:8090")
    ap.add_argument("--to", default=None)
    ap.add_argument("--amount", type=float, default=None)   # whole tokens
    ap.add_argument("--file", default=None)                 # submit-data: corpus file
    ap.add_argument("--media-type", default="text")
    ap.add_argument("--stake", type=float, default=None)    # whole tokens
    ap.add_argument("--data-id", default=None)              # challenge target
    ap.add_argument("--reason", default="validity")         # validity | ownership
    ap.add_argument("--challenge-id", default=None)         # vote target
    ap.add_argument("--support", action="store_true")       # vote: uphold challenge
    ap.add_argument("--passphrase-env", default=None,
                    help="non-interactive `new`: read the wallet passphrase from "
                         "this env var (unset/empty = unencrypted)")
    a = ap.parse_args()

    if a.cmd == "new":
        rec = create(a.path, a.passphrase_env)
        print(f"wallet created: {a.path}  (mode 0600 — BACK THIS FILE UP)")
        print(f"address: {to_display(rec['address'])}  (hex {rec['address']})")
        print(f"pubkey:  {rec['pub']}")
        return
    if a.cmd == "restore":
        restore(a.path)
        return

    key, rec = load(a.path)
    if a.cmd == "show":
        print(f"address: {to_display(rec['address'])}  (hex {rec['address']})")
        print(f"pubkey:  {rec['pub']}")
    elif a.cmd == "balance":
        out = _get(a.node, f"/balance?addr={rec['address']}")
        print(f"address: {to_display(rec['address'])}")
        print(f"balance: {out['grains'] / GRAIN:.9f} SESTRIAN "
              f"({out['grains']} grains) @ block {out['height']}")
    elif a.cmd == "send":
        if not a.to or a.amount is None:
            raise SystemExit("send needs --to and --amount")
        def _build(nonce):
            tx = TransferTx(from_pub=rec["pub"], to_addr=parse_addr(a.to),
                            amount=int(round(a.amount * GRAIN)),
                            nonce=nonce).signed(key)
            return tx, {"from_pub": tx.from_pub, "to_addr": tx.to_addr,
                        "amount": tx.amount, "nonce": tx.nonce,
                        "sig": tx.sig.hex()}
        out = _submit(a.node, rec["address"], "/transfer", _build)
        print(f"CONFIRMED: sent {a.amount} to {a.to}: {out}")
    elif a.cmd == "submit-data":
        if not a.file or a.stake is None:
            raise SystemExit("submit-data needs --file and --stake")
        import hashlib
        import os
        from rig.token import DataSubmitTx
        # stream the hash — corpora are routinely multi-GB and must never be
        # slurped into memory just to be fingerprinted
        h = hashlib.sha256()
        size = 0
        with open(a.file, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
                size += len(chunk)
        def _build(nonce):
            tx = DataSubmitTx(owner_pub=rec["pub"], data_hash=h.hexdigest(),
                              size_bytes=size, media_type=a.media_type,
                              stake=int(round(a.stake * GRAIN)),
                              nonce=nonce).signed(key)
            return tx, {"owner_pub": tx.owner_pub, "data_hash": tx.data_hash,
                        "size_bytes": tx.size_bytes, "media_type": tx.media_type,
                        "stake": tx.stake, "nonce": tx.nonce,
                        "sig": tx.sig.hex()}
        out = _submit(a.node, rec["address"], "/data/submit", _build)
        print(f"CONFIRMED: data staked ({size} bytes, hash {h.hexdigest()[:16]}…, "
              f"stake {a.stake}): {out}")
        print(f"  now mine with:  --data-refs {h.hexdigest()}")
    elif a.cmd == "challenge":
        if not a.data_id or a.stake is None:
            raise SystemExit("challenge needs --data-id and --stake")
        from rig.token import DataChallengeTx
        info = _get(a.node, f"/balance?addr={rec['address']}")
        tx = DataChallengeTx(challenger_pub=rec["pub"], data_id=a.data_id,
                             stake=int(round(a.stake * GRAIN)), reason=a.reason,
                             nonce=info.get("nonce", 0)).signed(key)
        out = _post(a.node, "/data/challenge", {
            "challenger_pub": tx.challenger_pub, "data_id": tx.data_id,
            "stake": tx.stake, "reason": tx.reason, "nonce": tx.nonce,
            "sig": tx.sig.hex()})
        print(f"challenge filed against {a.data_id[:12]} ({a.reason}): {out}")
    elif a.cmd == "vote":
        if not a.challenge_id:
            raise SystemExit("vote needs --challenge-id (and --support to uphold)")
        from rig.token import DataVoteTx
        info = _get(a.node, f"/balance?addr={rec['address']}")
        tx = DataVoteTx(voter_pub=rec["pub"], challenge_id=a.challenge_id,
                        support=a.support, nonce=info.get("nonce", 0)).signed(key)
        out = _post(a.node, "/data/vote", {
            "voter_pub": tx.voter_pub, "challenge_id": tx.challenge_id,
            "support": tx.support, "nonce": tx.nonce, "sig": tx.sig.hex()})
        print(f"vote {'FOR' if a.support else 'AGAINST'} challenge: {out}")
    elif a.cmd == "registry":
        out = _get(a.node, "/data/registry")
        for did, e in out["registry"].items():
            print(f"{did[:16]}  {e['status']:8} {e['media_type']:6} "
                  f"{e['size']:>10}B  stake {e['stake']/GRAIN:.2f}  owner {e['owner'][:12]}")
        for cid, c in out["challenges"].items():
            print(f"⚔ {cid[:16]} vs {c['data_id'][:12]} ({c['reason']}) "
                  f"expires h{c['expiry']} votes {len(c['votes_for'])}:{len(c['votes_against'])}")


if __name__ == "__main__":
    main()
