import { build } from 'esbuild';

const shared = {
  entryPoints: ['src/index.tsx'],
  platform: 'node',
  target: 'node20',
  jsx: 'automatic',
  logLevel: 'info',
};

// ESM build for `npm install -g govon` / `npx govon`
// Dependencies are external — resolved via node_modules at runtime
await build({
  ...shared,
  outfile: 'dist/index.js',
  format: 'esm',
  bundle: true,
  packages: 'external',
});

// NOTE: CJS/SEA bundle (dist/govon.cjs) will be added in Phase 5
// when the Node SEA build pipeline is implemented. Ink uses top-level
// await which is incompatible with CJS format — SEA will use ESM blob.

console.log('Build complete: dist/index.js (ESM)');
