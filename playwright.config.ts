import { defineConfig } from '@playwright/test';

/**
 * Playwright E2E 테스트 설정
 *
 * R1 기준 shell-first runtime contract를 우선 검증한다.
 * 현재 스위트는 request fixture 기반으로 runtime smoke/API contract를 검사하고,
 * 향후 /api/v2/* session runtime 및 shell transcript 시나리오를 같은 디렉터리에서 확장한다.
 */
export default defineConfig({
  testDir: './e2e',
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
