#!/usr/bin/env node
import('../dist/index.js').catch((err) => {
  console.error('Failed to start govon:', err.message);
  process.exit(1);
});
