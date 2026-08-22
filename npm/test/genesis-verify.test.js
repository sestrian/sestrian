'use strict';
// The genesis download is the one place the CLI takes 650MB of consensus state
// from the network and writes it to disk. Its checksums are the only thing
// standing between a tampered mirror and a node that silently runs a different
// chain — so they get a real test, with fetch stubbed and deliberately corrupt
// manifests, rather than being trusted because the happy path worked once.

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');

const { ensureGenesis } = require('../scripts/download.js');

const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');

// A stand-in for the real 650MB weight vector: the code under test only ever
// hashes and writes these bytes, so size is irrelevant to what we are checking.
const RAW = Buffer.from('these bytes stand in for the genesis weight vector');
// Compressed form is a literal rather than zlib.zstdCompressSync() output,
// because that API only exists on newer Node — and the older runtimes are
// exactly the ones where we need this test to exercise the zstd-CLI fallback.
const ZST = Buffer.from(
  'KLUv/SAykQEAdGhlc2UgYnl0ZXMgc3RhbmQgaW4gZm9yIHRoZSBnZW5lc2lzIHdlaWdodCB2ZWN0b3I=',
  'base64'
);

function manifest({ rawHash = sha(RAW), zstHash = sha(ZST) } = {}) {
  return {
    network: 'devnet',
    params: 12,
    raw: { bytes: RAW.length, sha256: rawHash },
    zstd: { file: 'genesis.bin.zst', bytes: ZST.length, sha256: zstHash },
  };
}

// Serve the manifest and the archive to whatever URL the downloader asks for.
function stubFetch(m) {
  global.fetch = async (url) => ({
    ok: true,
    status: 200,
    arrayBuffer: async () => Buffer.from(JSON.stringify(m)),
    get body() {
      return new ReadableStream({
        start(c) { c.enqueue(new Uint8Array(ZST)); c.close(); },
      });
    },
    _url: url,
  });
}

function tmpDest(name) {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'sestrian-genesis-test-'));
  return path.join(d, name);
}

test('accepts an archive whose weights hash to the advertised root', async () => {
  stubFetch(manifest());
  const dest = tmpDest('genesis.bin');
  await ensureGenesis(dest);
  assert.deepEqual(fs.readFileSync(dest), RAW, 'decompressed bytes must land verbatim');
});

test('rejects weights that do not match, and writes nothing', async () => {
  stubFetch(manifest({ rawHash: 'b'.repeat(64) }));
  const dest = tmpDest('genesis.bin');
  await assert.rejects(() => ensureGenesis(dest), /CHECKSUM MISMATCH \(weights\)/);
  assert.equal(fs.existsSync(dest), false, 'a rejected genesis must not be installed');
  assert.equal(fs.existsSync(`${dest}.part`), false, 'partial file must be cleaned up');
});

test('rejects a tampered archive before it is ever decompressed', async () => {
  stubFetch(manifest({ zstHash: 'c'.repeat(64) }));
  const dest = tmpDest('genesis.bin');
  await assert.rejects(() => ensureGenesis(dest), /CHECKSUM MISMATCH \(archive\)/);
  assert.equal(fs.existsSync(dest), false);
});

test('refuses a manifest with no checksums rather than trusting it', async () => {
  stubFetch({ network: 'devnet', zstd: { file: 'genesis.bin.zst', bytes: 1 } });
  await assert.rejects(() => ensureGenesis(tmpDest('genesis.bin')), /missing checksums/);
});

test('an existing genesis is reused, not re-downloaded', async () => {
  const dest = tmpDest('genesis.bin');
  fs.writeFileSync(dest, RAW);
  global.fetch = async () => { throw new Error('must not hit the network'); };
  assert.equal(await ensureGenesis(dest), dest);
});
