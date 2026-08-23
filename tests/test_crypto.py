"""Signed identity and transactions (§4, §5). Kept small — Ed25519 is the slow
reference implementation."""

from rig.crypto import BackpropTx, Key, delta_hash, verify


def test_sign_verify_roundtrip_and_tamper():
    k = Key.generate(b"seed-a".ljust(32, b"0"))
    msg = b"sestrian block 42"
    sig = k.sign(msg)
    assert verify(k.pub, msg, sig)
    assert not verify(k.pub, b"different message", sig)   # tamper -> invalid


def test_distinct_seeds_give_distinct_keys():
    a = Key.generate(b"seed-a".ljust(32, b"0"))
    b = Key.generate(b"seed-b".ljust(32, b"0"))
    assert a.pub != b.pub


def test_backprop_tx_signature():
    k = Key.generate(b"miner".ljust(32, b"0"))
    tx = BackpropTx(miner=k.pub, base_height=3,
                    delta_hash=delta_hash(b"BODY"), da_pointer="da://x").signed(k)
    assert tx.verify()
    assert len(tx.txid()) == 64


def test_forged_signature_rejected():
    k = Key.generate(b"miner".ljust(32, b"0"))
    attacker = Key.generate(b"attacker".ljust(32, b"0"))
    tx = BackpropTx(miner=k.pub, base_height=3,
                    delta_hash=delta_hash(b"BODY"), da_pointer="da://x")
    tx.sig = attacker.sign(tx.signing_bytes())        # wrong key signs
    assert not tx.verify()
