#!/usr/bin/env node
'use strict';

const { spawn } = require('child_process');
const { printEnvironmentStatus, isGovonInstalled } = require('../lib/python-check');

const args = process.argv.slice(2);

// If called via postinstall or directly with --check-install, run environment check only and exit
if (args[0] === '--check-install') {
  const ok = printEnvironmentStatus();
  process.exit(ok ? 0 : 1);
}

// Actual CLI execution path: print guidance and exit if environment is not ready
if (!printEnvironmentStatus()) {
  process.exit(1);
}

// Launch the govon CLI
const child = spawn('govon', args, {
  stdio: 'inherit',
  shell: false,
});

child.on('error', (err) => {
  if (err.code === 'ENOENT') {
    console.error(
      [
        '',
        '  [govon] Could not run the govon command.',
        '  Please verify it was installed via pip install govon.',
        '',
      ].join('\n')
    );
  } else {
    console.error(`[govon] Execution error: ${err.message}`);
  }
  process.exit(1);
});

child.on('close', (code) => {
  process.exit(code ?? 0);
});
