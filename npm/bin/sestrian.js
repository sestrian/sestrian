#!/usr/bin/env node
'use strict';
// Thin wrapper around the sestrian-node binary. It adds defaults that match
// scripts/install.sh (so the two paths agree) and a readable `status`; every
// other argument is handed straight to the node, which owns all real behaviour.

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { ensureBinary, HOME_DIR } = require('../scripts/download.js');

const API_PORT = process.env.SESTRIAN_API_PORT || '8090';
const DATA_DIR = path.join(HOME_DIR, 'nodedata');
const WALLET = path.join(HOME_DIR, 'wallet.json');
const GENESIS = path.join(HOME_DIR, 'genesis.bin');

const USAGE = `sestrian — run a node on the Sestrian devnet

  sestrian run       start a node (syncs and serves; add --produce to mine)
  sestrian check     preflight: can this machine actually contribute?
  sestrian status    query a node already running on :${API_PORT}
  sestrian <flags>   anything else is passed to sestrian-node unchanged
                     (sestrian node-help lists the node's own flags —
                      that one downloads the binary first)

Defaults, matching scripts/install.sh:
  --data-dir ${DATA_DIR}
  --wallet ${WALLET}        (if present)
  --genesis-file ${GENESIS}     (if present)

This package ships the NODE only. Mining also needs the Python trainer —
see https://github.com/sestrian/sestrian/blob/main/docs/joining.md
`;

// Only pass paths that exist: the node prints genuinely useful guidance when a
// genesis or identity is missing, and clobbering that with "file unreadable"
// would be a worse error than the one it already writes.
function defaults() {
  const a = ['--data-dir', DATA_DIR];
  if (fs.existsSync(WALLET)) a.push('--wallet', WALLET);
  if (fs.existsSync(GENESIS)) a.push('--genesis-file', GENESIS);
  a.push('--api-port', String(API_PORT));
  return a;
}

function runNode(bin, args) {
  const child = spawn(bin, args, { stdio: 'inherit' });
  child.on('error', (e) => {
    console.error(`sestrian: could not execute ${bin}: ${e.message}`);
    process.exit(1);
  });
  child.on('exit', (code, signal) => process.exit(signal ? 1 : (code ?? 0)));
}

async function status() {
  const url = `http://127.0.0.1:${API_PORT}/status`;
  let res;
  try {
    res = await fetch(url, { signal: AbortSignal.timeout(5000) });
  } catch {
    console.error(
      `no node responding on 127.0.0.1:${API_PORT}.\n` +
      `Start one with:  sestrian run\n` +
      `(or set SESTRIAN_API_PORT if yours listens elsewhere)`
    );
    process.exit(1);
  }
  if (!res.ok) {
    console.error(`node returned HTTP ${res.status} from ${url}`);
    process.exit(1);
  }

  const s = await res.json();
  const row = (k, v) => console.log(`  ${String(k).padEnd(16)} ${v}`);
  console.log('');
  row('height', s.height);
  row('peers', s.peers);
  row('supply', `${(Number(s.supply || 0) / 1e9).toFixed(3)} SESTRIAN`);
  row('producing', s.producer ? 'yes' : 'no');
  row('trainer', s.model_attached ? 'attached' : 'not attached');
  row('stale_deltas', s.stale_deltas);
  console.log('');

  if (Number(s.stale_deltas) > 0) {
    console.log(
      `WARNING: ${s.stale_deltas} consecutive deltas were dropped as stale.\n` +
      `Your training rounds are finishing after the head moves on, so the work\n` +
      `is discarded and you are EARNING NOTHING. Lower --inner on the trainer.\n`
    );
    process.exit(2);
  }
  if (Number(s.peers) === 0) {
    console.log('WARNING: no peers connected — this node is isolated.\n');
  }
}

async function main() {
  const [cmd, ...rest] = process.argv.slice(2);

  // Help must never trigger a download: `npx sestrian --help` on a fresh machine
  // has to print usage, not fail trying to fetch a binary it doesn't need yet.
  if (!cmd || cmd === 'help' || cmd === '--help' || cmd === '-h') {
    return void console.log(USAGE);
  }
  if (cmd === 'status') return void (await status());

  const bin = await ensureBinary();
  // explicit escape hatch for the node's own --help, which does need the binary
  if (cmd === 'node-help') return runNode(bin, ['--help']);
  if (cmd === 'run') return runNode(bin, [...defaults(), ...rest]);
  if (cmd === 'check') return runNode(bin, ['--check', ...defaults(), ...rest]);
  return runNode(bin, [cmd, ...rest]); // straight through to the node
}

main().catch((e) => {
  console.error(`sestrian: ${e.message}`);
  process.exit(1);
});
