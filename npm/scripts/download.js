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

module.exports = { ensureBinary, targetTriple, HOME_DIR, BIN_NAME, TAGS };
