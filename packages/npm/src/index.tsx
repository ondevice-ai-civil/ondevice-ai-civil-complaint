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
