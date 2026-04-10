import { build } from 'esbuild';

const shared = {
  platform: 'node',
  target: 'node20',
  jsx: 'automatic',
  logLevel: 'info',
  format: 'esm',
  bundle: true,
  packages: 'external',
};

// ESM build for `npm install -g govon` / `npx govon` (Ink TUI)
await build({
  ...shared,
  entryPoints: ['src/index.tsx'],
  outfile: 'dist/index.js',
});

// ESM build for `govon-repl` — REPL-style TUI prototype (no React/Ink)
await build({
  ...shared,
  entryPoints: ['src/repl.ts'],
  outfile: 'dist/repl.js',
});

// NOTE: CJS/SEA bundle (dist/govon.cjs) will be added in Phase 5
// when the Node SEA build pipeline is implemented. Ink uses top-level
// await which is incompatible with CJS format — SEA will use ESM blob.

console.log('Build complete: dist/index.js (ESM), dist/repl.js (ESM)');
