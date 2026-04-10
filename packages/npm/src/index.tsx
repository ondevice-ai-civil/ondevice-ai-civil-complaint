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
import { RenderInterceptor } from './terminal/index.js';
import { TerminalSizeProvider } from './contexts/index.js';

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
// Rendering engine — alternate screen + deferred erase + synchronized output
// ---------------------------------------------------------------------------
// Ink's log-update tracks previousLineCount to erase old output. On resize,
// text reflows so the visual row count changes but previousLineCount is stale,
// causing ghost artifacts. This is a known Ink limitation (PR #916).
//
// Solution (adapted from Claude Code's ink.tsx):
//   1. Alternate screen buffer — separate canvas with no scrollback
//   2. Deferred erase — resize sets a flag, erase happens inside next render
//   3. BSU/ESU synchronized output — entire frame written atomically
//
// The RenderInterceptor provides a proxy stdout stream that captures Ink's
// output and flushes it to the real terminal with these protections applied.
// ---------------------------------------------------------------------------
const interceptor = new RenderInterceptor(process.stdout);
interceptor.enterAltScreen();

const instance = render(
  <TerminalSizeProvider>
    <App version={pkg.version} initialQuery={query} />
  </TerminalSizeProvider>,
  { stdout: interceptor.stdout as unknown as NodeJS.WriteStream },
);

// Restore normal screen on exit (covers all exit paths)
process.on('exit', () => interceptor.destroy());

// On resize: deferred erase pattern — sets flag, Ink re-renders, next flush
// prepends ERASE_SCREEN + CURSOR_HOME inside BSU/ESU atomic write.
process.stdout.on('resize', () => interceptor.handleResize());
