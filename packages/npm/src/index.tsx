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

// ---------------------------------------------------------------------------
// Alternate screen buffer — eliminates resize ghost lines
// ---------------------------------------------------------------------------
// Ink's log-update tracks previousLineCount to erase old output. On resize,
// text reflows so the visual row count changes but previousLineCount is stale,
// causing ghost artifacts. This is a known Ink limitation (PR #916).
//
// Solution: enter the alternate screen buffer (\x1b[?1049h), the same
// mechanism used by vim, htop, and Claude Code. The alternate screen is a
// separate canvas with NO scrollback, so clearing it on resize is safe —
// no scrollback pollution. On exit, the original screen is restored.
// ---------------------------------------------------------------------------
process.stdout.write('\x1b[?1049h\x1b[2J\x1b[H');

const instance = render(<App version={pkg.version} initialQuery={query} />);

// Restore normal screen on exit (covers all exit paths)
const restoreScreen = () => {
  process.stdout.write('\x1b[?1049l');
};
process.on('exit', restoreScreen);

// On resize: clear the alternate screen canvas and reset log-update state,
// then let Ink's own handler recalculate layout and re-render cleanly.
process.stdout.prependListener('resize', () => {
  process.stdout.write('\x1b[2J\x1b[H');
  instance.clear();
});
