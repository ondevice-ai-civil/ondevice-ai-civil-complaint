#!/usr/bin/env node
'use strict';

const { spawn } = require('child_process');
const { printEnvironmentStatus, isGovonInstalled } = require('../lib/python-check');

const args = process.argv.slice(2);

// On postinstall or direct invocation with --check-install, only run environment check and exit
if (args[0] === '--check-install') {
  const ok = printEnvironmentStatus();
  process.exit(ok ? 0 : 1);
}

// Actual CLI execution path: if environment is not valid, print guidance and exit
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
        '  [govon] Unable to execute the govon command.',
        '  Please verify it is installed via: pip install govon',
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
