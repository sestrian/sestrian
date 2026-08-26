"""Drive the bridge's real `run()` loop over a real socket.

Two production outages came out of this function and neither was reachable by
the suite, because nothing ever called it:

  * `torch` is imported lazily inside functions here, so a later `import torch`
    in run() makes the name function-local. Using it earlier raised
    UnboundLocalError on start and took the GPU miner's mining down.
  * the chat sampler was handed a generator built on the wrong device.

The first is caught simply by running the loop at all. The second is caught by
asserting how the generator is CONSTRUCTED rather than comparing devices after
the fact — comparing them cannot work on a CPU-only machine, since a hardcoded
`torch.Generator()` is already a cpu generator and the comparison passes. That
was not a guess: the compare-based version of the test was written first, the
bug was reintroduced, and it went green.

All three failure modes were re-introduced deliberately to confirm these tests
fail on them: the unbound `torch`, the device-less generator, and sampling from
the global RNG.

The bridge is driven exactly as the node drives it: a listening socket, the
same 4-byte length framing, the same message order.
"""

import json
import socket
import struct
import threading

import numpy as np
import pytest

from client import miner_bridge
from client.gossip import MODEL_PRESETS
from client.moe import MoEGPT

MODEL = "toy-moe"          # 332,416 params — small enough to build in a test


def _dim(name: str) -> int:
    return sum(p.numel() for p in MoEGPT(MODEL_PRESETS[name]).parameters())


def _send(sock, obj):
    raw = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _send_bin(sock, raw: bytes):
    sock.sendall(struct.pack(">I", len(raw)) + raw)


def _recv(sock) -> dict:
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            raise ConnectionError("bridge closed")
        hdr += chunk
    n = struct.unpack(">I", hdr)[0]
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("bridge closed")
        buf += chunk
    return json.loads(bytes(buf))


class _Args:
    """Stands in for the argparse Namespace run() reads."""

    def __init__(self, port):
        self.node_port = port
        self.model = MODEL
        self.device = "cpu"
        self.serve_only = True     # never train: this test is about the loop
        self.inner = 1
        self.batch = 2
        self.data = None


@pytest.fixture()
def bridge():
    """A live bridge connected to us, with the model already synced."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    err = {}

    def _run():
        try:
            miner_bridge.run(_Args(port))
        except BaseException as e:            # noqa: BLE001 - surfaced below
            err["e"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    srv.settimeout(30)
    conn, _ = srv.accept()
    conn.settimeout(60)

    hello = _recv(conn)
    assert hello["t"] == "hello", f"expected hello, got {hello} (bridge err: {err})"

    # The node names the model and ships the chain state as a raw i64 frame.
    _send(conn, {"t": "state", "height": 7, "model": MODEL})
    _send_bin(conn, np.zeros(_dim(MODEL), dtype="<i8").tobytes())

    yield conn, err

    conn.close()
    srv.close()


def test_run_loop_starts_and_serves_a_generation(bridge):
    """The regression for the UnboundLocalError: reaching this at all means
    run() got past its own imports and through a full state sync."""
    conn, err = bridge
    _send(conn, {"t": "generate", "prompt": "hello", "n": 4})
    reply = _recv(conn)
    assert not err, f"bridge raised: {err.get('e')!r}"
    assert reply["t"] == "generated"
    assert reply["height"] == 7
    assert isinstance(reply["text"], str)


def test_unsynced_bridge_answers_rather_than_hanging(bridge):
    """A second generation must also come back — the first must not wedge the
    loop, which is the shape of the bug that left chat permanently busy."""
    conn, err = bridge
    for _ in range(3):
        _send(conn, {"t": "generate", "prompt": "again", "n": 2})
        reply = _recv(conn)
        assert reply["t"] == "generated", f"bridge err: {err.get('e')!r}"


def test_chat_generator_is_built_for_the_tensor_device(monkeypatch):
    """The CUDA regression, caught on a machine with no CUDA.

    torch.multinomial rejects a generator whose device differs from the
    tensor's, and that mismatch crash-looped the GPU miner. Comparing the two
    devices after the fact CANNOT catch it on CPU: a hardcoded
    `torch.Generator()` is already a cpu generator, so the comparison passes and
    the bug ships anyway. Verified that by reintroducing it — the device-compare
    version of this test went green.

    So assert the CONSTRUCTION instead: the generator must be built with an
    explicit device taken from the tensors. That is false for `torch.Generator()`
    and for None on any machine, GPU or not.
    """
    import torch

    built = {}
    real_gen = torch.Generator

    def spy_generator(*args, **kwargs):
        built["device"] = kwargs.get("device", args[0] if args else None)
        return real_gen(*args, **kwargs)

    seen = {}
    real_mult = torch.multinomial

    def spy_multinomial(probs, num_samples, *args, **kwargs):
        seen["probs_device"] = probs.device
        seen["gen"] = kwargs.get("generator")
        return real_mult(probs, num_samples, *args, **kwargs)

    monkeypatch.setattr(torch, "Generator", spy_generator)
    monkeypatch.setattr(torch, "multinomial", spy_multinomial)

    model = MoEGPT(MODEL_PRESETS[MODEL]).eval()

    class _Sock:
        def sendall(self, raw):
            pass

    miner_bridge._serve_generate(
        _Sock(), {"prompt": "hi", "n": 2}, model, "cpu", 7,
        np.zeros(_dim(MODEL), dtype="<i8"))

    assert seen.get("gen") is not None, (
        "generation sampled from the GLOBAL rng — interleaved with training that "
        "makes a miner's delta unreproducible, and unreproducible is slashable")
    assert built.get("device") is not None, (
        "the chat generator was built with no device. torch.multinomial rejects a "
        "CPU generator against CUDA tensors, so this passes on a laptop and "
        "crash-loops the GPU miner — which is exactly what it did.")
    assert str(built["device"]) == str(seen["probs_device"]), (
        f"generator built for {built['device']} but tensors are on "
        f"{seen['probs_device']}")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason="no CUDA device on this machine")
def test_generation_runs_on_cuda():
    """The real thing, on a real GPU. Skipped in CI, which has no card — run it
    on the miner: `pytest tests/test_miner_bridge_loop.py -k cuda`.

    A skipped test protects nothing, so this exists to be run deliberately on
    the box that actually has the hardware, not to make CI look green.
    """
    import torch

    model = MoEGPT(MODEL_PRESETS[MODEL]).cuda().eval()
    sent = {}

    class _Sock:
        def sendall(self, raw):
            sent["raw"] = raw

    miner_bridge._serve_generate(
        _Sock(), {"prompt": "hi", "n": 4}, model, "cuda", 11,
        np.zeros(_dim(MODEL), dtype="<i8"))

    assert sent["raw"], "nothing was sent back"
    body = json.loads(sent["raw"][4:])
    assert body["t"] == "generated" and body["height"] == 11
