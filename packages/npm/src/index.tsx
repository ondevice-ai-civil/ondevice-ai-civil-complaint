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

render(<App version={pkg.version} initialQuery={query} />);

// ---------------------------------------------------------------------------
// Fix terminal resize ghost lines
// ---------------------------------------------------------------------------
// Ink's log-update tracks `previousLineCount` by counting `\n` in the last
// output. On resize, text reflows so the actual visual row count changes, but
// previousLineCount stays stale. `eraseLines(previousLineCount)` then erases
// the wrong number of rows, leaving ghost artifacts.
//
// Fix: access the Ink instance via its internal WeakMap (bypassing the package
// exports restriction with createRequire), remove Ink's broken resize handler,
// and replace it with one that clears the visible screen before re-rendering.
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const inkInstances = require('ink/build/instances.js').default as WeakMap<typeof process.stdout, any>;
const ink = inkInstances.get(process.stdout);

if (ink) {
  // Remove Ink's default resize handler (uses stale previousLineCount)
  process.stdout.off('resize', ink.resized);

  process.stdout.on('resize', () => {
    // 1. Clear the visible screen (preserves scrollback) and home cursor.
    //    \x1b[2J  = erase entire visible screen
    //    \x1b[H   = move cursor to row 1, col 1
    process.stdout.write('\x1b[2J\x1b[H');

    // 2. Re-write all accumulated static output (chat history) so it remains
    //    visible without scrolling — mirrors Ink's own overflow path.
    if (ink.fullStaticOutput) {
      process.stdout.write(ink.fullStaticOutput);
    }

    // 3. Reset log-update internal state. After \x1b[H the cursor is at row 1,
    //    so the stale eraseLines inside clear() is harmless (can't go above row 1).
    //    This resets previousLineCount to 0 and previousOutput to ''.
    ink.log.clear();

    // 4. Force Ink to treat the next render as new output.
    ink.lastOutput = '';

    // 5. Recalculate Yoga layout at new terminal width and re-render.
    ink.calculateLayout();
    ink.onRender();
  });
}
