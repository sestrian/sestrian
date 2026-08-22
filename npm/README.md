# sestrian

Run a node on the [Sestrian](https://sestrian.com) devnet — a blockchain whose
state is the weights of a single public neural network.

```bash
npx sestrian check     # can this machine contribute?
npx sestrian run       # sync and serve
npx sestrian status    # how is my node doing?
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

The **genesis** (the network's starting weights, ~650MB) is not bundled either.
Download or reproduce it as described in the joining guide; the node verifies
whatever it is given against the state root compiled into the binary, so a
tampered genesis fails at startup rather than silently forking you.

## How the binary gets here

On first real use — not at install time — the CLI downloads the prebuilt
`sestrian-node` for your platform from GitHub Releases, checks its sha256
against the release's `SHA256SUMS`, and only then extracts it to
`~/.sestrian/bin/<tag>/`. A mismatch is a hard failure that prints the URL and
both hashes; nothing unverified is ever executed.

Installing lazily rather than in a `postinstall` hook keeps `npm install` offline-
and CI-safe, and means `npx sestrian --help` doesn't pay for a download.

Supported: linux x64/arm64, macOS arm64/x64, Windows x64.

| variable | effect |
|---|---|
| `SESTRIAN_RELEASE_TAG` | pin a specific release tag |
| `SESTRIAN_NODE_BIN` | use a local binary; skips the download entirely |
| `SESTRIAN_HOME` | data/wallet/binary root (default `~/.sestrian`) |
| `SESTRIAN_API_PORT` | API port for `status` (default `8090`) |

## Back up your wallet

`~/.sestrian/wallet.json` is your identity **and** your balance. There is no
recovery service.

MIT
