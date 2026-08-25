# sestrian

Run a node on the [Sestrian](https://sestrian.com) devnet — a blockchain whose
state is the weights of a single public neural network.

```bash
npx sestrian run       # sync and serve — sets everything up on first run
npx sestrian status    # how is my node doing?
npx sestrian check     # can this machine contribute?
npx sestrian genesis   # download + verify the genesis weights only
```

Any other arguments go straight to the node binary, so `npx sestrian --help`
shows the real flags.

## What this package is, and is not

It ships **the node only** — the Rust binary that validates the chain, syncs, and
serves inference. That covers watching, syncing and serving.

**It cannot mine on its own.** Training additionally needs the PyTorch trainer,
which is Python and far too large to bundle here. To mine, follow
[docs/joining.md](https://github.com/sestrian/sestrian/blob/main/docs/joining.md)
— `scripts/install.sh --mine` sets up both halves.

The **genesis** (the network's starting weights, 652MB) is not bundled either —
it is fetched on first `run`, or on demand with `npx sestrian genesis`.

## How the binary gets here

On first real use — not at install time — the CLI downloads the prebuilt
`sestrian-node` for your platform from GitHub Releases, checks its sha256
against the release's `SHA256SUMS`, and only then extracts it to
`~/.sestrian/bin/<tag>/`. A mismatch is a hard failure that prints the URL and
both hashes; nothing unverified is ever executed.

Installing lazily rather than in a `postinstall` hook keeps `npm install` offline-
and CI-safe, and means `npx sestrian --help` doesn't pay for a download.

The genesis is checked at two layers, and it is worth being exact about which
one proves what. The CLI hashes the archive and then the decompressed weights
against the manifest — that proves the **download is intact**, and nothing more.
Under protocol v2 the chain's genesis id is a page-merkle root over the model's
page table, not a flat hash of the file, so a matching file hash does not by
itself mean you hold the right chain. **The node establishes identity**: at
startup it recomputes the state root and refuses to run unless it matches the id
compiled into the binary. A tampered manifest gets you a file that dies there.

Either hash mismatch deletes the download and refuses. It is streamed and
decompressed on the fly, so the weights never sit in memory whole, and written
through a `.part` file — an interrupted download can never be mistaken for a
complete one.

Your **wallet** is created by the node on first run if you do not have one, in
the same format the Python client reads. Set `SESTRIAN_WALLET_PASSPHRASE` to
encrypt it; without one it is written unencrypted with `0600` permissions and
says so.

Supported: linux x64/arm64, macOS arm64/x64, Windows x64.

| variable | effect |
|---|---|
| `SESTRIAN_RELEASE_TAG` | pin a specific release tag |
| `SESTRIAN_NODE_BIN` | use a local binary; skips the download entirely |
| `SESTRIAN_HOME` | data/wallet/binary root (default `~/.sestrian`) |
| `SESTRIAN_API_PORT` | API port for `status` (default `8090`) |
| `SESTRIAN_GENESIS_TAG` | release tag to fetch the genesis from |
| `SESTRIAN_WALLET_PASSPHRASE` | encrypt a newly created wallet |

## Back up your wallet

`~/.sestrian/wallet.json` is your identity **and** your balance. There is no
recovery service.

MIT
