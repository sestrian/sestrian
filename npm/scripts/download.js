'use strict';
// Fetch + verify the prebuilt sestrian-node binary for this platform.
//
// LAZY, not postinstall: the download happens the first time you actually run
// `sestrian`, not at install time. A postinstall that reaches the network breaks
// offline installs, air-gapped CI, and `npm ci` behind a proxy — and it makes
// `npx sestrian --help` pay for a 10MB download it may not need. The cost is one
// visible download on first real use, which is the moment it makes sense.
//
// Nothing runs unverified: the archive's sha256 must match the release's
// SHA256SUMS or we delete it and refuse.

const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const REPO = process.env.SESTRIAN_REPO || 'sestrian/sestrian';
const VERSION = require('../package.json').version;

// Pinned tag first, then the rolling build. Explicit override always wins.
const TAGS = process.env.SESTRIAN_RELEASE_TAG
  ? [process.env.SESTRIAN_RELEASE_TAG]
  : [`v${VERSION}`, 'devnet-latest'];

const TARGETS = {
  'linux-x64': 'x86_64-unknown-linux-gnu',
  'linux-arm64': 'aarch64-unknown-linux-gnu',
  'darwin-arm64': 'aarch64-apple-darwin',
  'darwin-x64': 'x86_64-apple-darwin',
  'win32-x64': 'x86_64-pc-windows-msvc',
};

const HOME_DIR = process.env.SESTRIAN_HOME || path.join(os.homedir(), '.sestrian');
const BIN_NAME = process.platform === 'win32' ? 'sestrian-node.exe' : 'sestrian-node';

function targetTriple() {
  const key = `${process.platform}-${process.arch}`;
  const triple = TARGETS[key];
  if (!triple) {
    throw new Error(
      `no prebuilt sestrian-node for ${key}.\n` +
      `Supported: ${Object.keys(TARGETS).join(', ')}\n` +
      `Build from source instead: https://github.com/${REPO}/blob/main/docs/joining.md`
    );
  }
  return triple;
}

const assetName = (triple) =>
  `sestrian-node-${triple}.${process.platform === 'win32' ? 'zip' : 'tar.gz'}`;

const dl = (tag, file) =>
  `https://github.com/${REPO}/releases/download/${tag}/${file}`;

async function fetchBuffer(url) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    const err = new Error(`GET ${url} -> HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return Buffer.from(await res.arrayBuffer());
}

// The release's SHA256SUMS, as { filename: hash }.
async function checksums(tag) {
  const text = (await fetchBuffer(dl(tag, 'SHA256SUMS'))).toString('utf8');
  const map = {};
  for (const line of text.split('\n')) {
    const m = line.trim().match(/^([0-9a-f]{64})\s+\*?(.+)$/i);
    if (m) map[path.basename(m[2])] = m[1].toLowerCase();
  }
  return map;
}

function extract(archive, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  // `tar` reads .tar.gz everywhere and .zip via bsdtar (macOS, and Windows 10+
  // ships bsdtar as tar.exe). Keeps this dependency-free.
  execFileSync('tar', ['-xf', archive, '-C', destDir], { stdio: 'inherit' });
}

/**
 * Path to a verified local binary, downloading it on first use.
 * @returns {Promise<string>}
 */
async function ensureBinary() {
  if (process.env.SESTRIAN_NODE_BIN) return process.env.SESTRIAN_NODE_BIN; // dev escape hatch

  const triple = targetTriple();
  const asset = assetName(triple);

  let lastErr;
  for (const tag of TAGS) {
    const installed = path.join(HOME_DIR, 'bin', tag, BIN_NAME);
    if (fs.existsSync(installed)) return installed;

    try {
      process.stderr.write(`sestrian: fetching ${asset} from ${tag}\n`);
      const sums = await checksums(tag);
      const want = sums[asset];
      if (!want) {
        throw new Error(
          `release ${tag} has no checksum for ${asset}\n` +
          `  it lists: ${Object.keys(sums).join(', ') || '(nothing)'}`
        );
      }

      const url = dl(tag, asset);
      const buf = await fetchBuffer(url);
      const got = crypto.createHash('sha256').update(buf).digest('hex');
      if (got !== want) {
        throw new Error(
          `CHECKSUM MISMATCH — refusing to install.\n` +
          `  url:      ${url}\n` +
          `  expected: ${want}\n` +
          `  got:      ${got}`
        );
      }

      const staging = fs.mkdtempSync(path.join(os.tmpdir(), 'sestrian-'));
      const archive = path.join(staging, asset);
      fs.writeFileSync(archive, buf);
      extract(archive, path.dirname(installed));
      fs.rmSync(staging, { recursive: true, force: true });

      if (!fs.existsSync(installed)) {
        throw new Error(`archive ${asset} did not contain ${BIN_NAME}`);
      }
      if (process.platform !== 'win32') fs.chmodSync(installed, 0o755);
      process.stderr.write(`sestrian: verified sha256 ${got}\n`);
      return installed;
    } catch (e) {
      lastErr = e;
      // A missing pinned tag is expected before the first tagged release; fall
      // through to the rolling build. Anything else is worth showing now.
      if (e.status !== 404) process.stderr.write(`sestrian: ${e.message}\n`);
    }
  }

  throw new Error(
    `could not obtain sestrian-node for ${targetTriple()}.\n` +
    `  tried tags: ${TAGS.join(', ')}\n` +
    `  last error: ${lastErr && lastErr.message}\n` +
    `Override with SESTRIAN_RELEASE_TAG=<tag>, point SESTRIAN_NODE_BIN at a local\n` +
    `build, or build from source: https://github.com/${REPO}/blob/main/docs/joining.md`
  );
}

// ---------------------------------------------------------------------------
// Genesis
//
// The chain's state IS the model, so a node cannot validate anything without
// the 683MB genesis weight vector. Reproducing it locally is trustless but
// costs a PyTorch install; downloading it is one hash-verified fetch. We do the
// fetch and verify it hard, because the decompressed bytes hash to exactly the
// genesis state root — checking the download IS checking the chain identity,
// not merely that a file arrived intact.

const GENESIS_TAG = process.env.SESTRIAN_GENESIS_TAG || 'devnet-genesis-1';

function human(n) {
  return n >= 1 << 30 ? `${(n / 2 ** 30).toFixed(2)}GB` : `${Math.round(n / 2 ** 20)}MB`;
}

// A zstd decompressor, or null if this Node cannot do it natively.
// Streaming matters: the output is 683MB and must never be a single Buffer.
function zstdStream() {
  const zlib = require('zlib');
  return typeof zlib.createZstdDecompress === 'function'
    ? zlib.createZstdDecompress()
    : null;
}

function hasZstdCli() {
  try {
    execFileSync('zstd', ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Path to a verified genesis.bin, downloading it if absent.
 * @returns {Promise<string>}
 */
async function ensureGenesis(dest) {
  if (fs.existsSync(dest)) return dest;

  const stream = require('stream');
  const { pipeline } = require('stream/promises');

  const manifest = JSON.parse(
    (await fetchBuffer(dl(GENESIS_TAG, 'genesis-manifest.json'))).toString('utf8')
  );
  const wantZst = manifest.zstd?.sha256;
  const wantRaw = manifest.raw?.sha256;
  if (!wantZst || !wantRaw) throw new Error(`genesis manifest is missing checksums`);

  const asset = manifest.zstd.file || 'genesis.bin.zst';
  const url = dl(GENESIS_TAG, asset);
  process.stderr.write(
    `sestrian: fetching genesis ${human(manifest.zstd.bytes)} -> ` +
    `${human(manifest.raw.bytes)}` +
    (manifest.params ? ` (${manifest.params.toLocaleString()} parameters)` : '') + `\n` +
    `sestrian: this is a one-time download; the model is the chain state\n`
  );

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const part = `${dest}.part`;
  const zstPart = `${dest}.zst.part`;
  const cleanup = () => {
    for (const f of [part, zstPart]) { try { fs.unlinkSync(f); } catch {} }
  };

  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);

  const zstSum = crypto.createHash('sha256');
  const rawSum = crypto.createHash('sha256');
  let seen = 0;
  let lastPct = -1;
  const total = manifest.zstd.bytes;
  const watch = new stream.Transform({
    transform(chunk, _enc, cb) {
      zstSum.update(chunk);
      seen += chunk.length;
      // A carriage return only redraws on a terminal. Piped to a log or a CI
      // job it just concatenates every tick into one unreadable line, so there
      // we print sparse, newline-terminated milestones instead.
      const tty = process.stderr.isTTY === true;
      const step = tty ? 5 : 25;
      const pct = Math.floor((seen / total) * 100);
      if (pct !== lastPct && pct % step === 0) {
        lastPct = pct;
        process.stderr.write(tty
          ? `\rsestrian: downloading genesis ${pct}%`
          : `sestrian: downloading genesis ${pct}%\n`);
      }
      cb(null, chunk);
    },
  });
  const tally = new stream.Transform({
    transform(chunk, _enc, cb) { rawSum.update(chunk); cb(null, chunk); },
  });

  try {
    const unzstd = zstdStream();
    if (unzstd) {
      await pipeline(
        stream.Readable.fromWeb(res.body), watch, unzstd, tally, fs.createWriteStream(part)
      );
    } else if (hasZstdCli()) {
      // Older Node: land the archive, then let the system zstd expand it.
      await pipeline(stream.Readable.fromWeb(res.body), watch, fs.createWriteStream(zstPart));
      execFileSync('zstd', ['-d', '-f', '-o', part, zstPart], { stdio: 'inherit' });
      await pipeline(fs.createReadStream(part), tally, new stream.Writable({
        write(_c, _e, cb) { cb(); },
      }));
      fs.unlinkSync(zstPart);
    } else {
      throw new Error(
        `cannot decompress zstd: this Node (${process.version}) has no built-in zstd ` +
        `and no \`zstd\` binary is on PATH.\n` +
        `  Install zstd (brew install zstd / apt install zstd) and retry, or fetch it yourself:\n` +
        `    curl -fL -o genesis.bin.zst ${url}\n` +
        `    zstd -d genesis.bin.zst -o ${dest}`
      );
    }
    if (process.stderr.isTTY === true) process.stderr.write('\n');

    const gotZst = zstSum.digest('hex');
    const gotRaw = rawSum.digest('hex');
    if (gotZst !== wantZst) {
      throw new Error(
        `GENESIS CHECKSUM MISMATCH (archive) — refusing to install.\n` +
        `  expected: ${wantZst}\n  got:      ${gotZst}`
      );
    }
    // The decisive check: these bytes hash to the genesis state root, so a pass
    // here means we hold the same chain everyone else does.
    if (gotRaw !== wantRaw) {
      throw new Error(
        `GENESIS CHECKSUM MISMATCH (weights) — refusing to install.\n` +
        `  expected: ${wantRaw}\n  got:      ${gotRaw}`
      );
    }
    fs.renameSync(part, dest);
    process.stderr.write(`sestrian: verified genesis state_root ${gotRaw}\n`);
    return dest;
  } catch (e) {
    cleanup();
    throw e;
  }
}

module.exports = { ensureBinary, ensureGenesis, targetTriple, HOME_DIR, BIN_NAME, TAGS };
