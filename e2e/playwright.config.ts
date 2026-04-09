import { defineConfig } from '@playwright/test';

/**
 * Playwright E2E test configuration
 *
 * Prioritizes validation of the shell-first runtime contract based on the R1 milestone.
 * The current suite verifies runtime smoke and API contracts using the request fixture.
 * Future /api/v2/* session runtime and shell transcript scenarios will be added in the same directory.
 */
export default defineConfig({
  testDir: '.',
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  outputDir: 'test-results/',
  fullyParallel: true,
  workers: process.env.CI ? 1 : undefined,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI
    ? [['github'], ['html', { open: 'never' }]]
    : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.GOVON_RUNTIME_BASE_URL ?? 'http://127.0.0.1:8000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'on-first-retry',
  },
});
