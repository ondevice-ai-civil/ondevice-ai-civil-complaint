/**
 * index.tsx — CLI entry point for the GovOn npm TUI.
 *
 * Parses CLI arguments with yargs and renders the <App /> component via Ink.
 */

import React from 'react';
import { render } from 'ink';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { createRequire } from 'node:module';
import { App } from './App.js';

// Read version from package.json at runtime (avoid bundling the whole file)
const require = createRequire(import.meta.url);
const pkg = require('../package.json') as { version: string };

const argv = yargs(hideBin(process.argv))
  .scriptName('govon')
  .usage('$0 [query]', 'GovOn agentic TUI for civil complaint assistance')
  .positional('query', {
    type: 'string',
    describe: 'Initial query to submit immediately',
  })
  .option('version', {
    alias: 'v',
    type: 'boolean',
    describe: 'Show version',
  })
  .help()
  .parseSync();

if (argv.version) {
  process.stdout.write(pkg.version + '\n');
  process.exit(0);
}

// argv._[0] is the positional <query> argument when provided
const query = argv._[0] as string | undefined;

const instance = render(<App version={pkg.version} initialQuery={query} />);

// ---------------------------------------------------------------------------
// Fix terminal resize ghost lines
// ---------------------------------------------------------------------------
// Ink's log-update tracks `previousLineCount` by counting `\n` in the last
// output. On resize, text reflows so the actual visual row count changes, but
// previousLineCount stays stale → eraseLines() erases wrong number of rows →
// ghost artifacts.
//
// Fix: use prependListener to fire BEFORE Ink's own resize handler.
// 1. Clear the visible screen (\x1b[2J preserves scrollback, unlike \x1b[3J)
// 2. Home cursor (\x1b[H)
// 3. Reset log-update internals via instance.clear() — the eraseLines() inside
//    clear() is harmless because the screen is already blank and cursor is at
//    row 1 (can't move above row 1). This resets previousLineCount=0 and
//    previousOutput='', so Ink's subsequent re-render writes fresh output.
// 4. Ink's own resize handler then fires: calculateLayout() + onRender() with
//    correct state, rendering cleanly at the new terminal width.
// ---------------------------------------------------------------------------
process.stdout.prependListener('resize', () => {
  process.stdout.write('\x1b[2J\x1b[H');
  instance.clear();
});
