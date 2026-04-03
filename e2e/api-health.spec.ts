import { expect, test } from '@playwright/test';

const apiKey = process.env.GOVON_API_KEY ?? 'dev-e2e-key';

const searchPayload = {
  query: '도로 보수 요청',
  doc_type: 'case',
  top_k: 3,
};

test.describe('GovOn Runtime Smoke', () => {
  test('/health returns runtime metadata for shell clients', async ({ request }) => {
    const response = await request.get('/health');

    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('status', 'healthy');
    expect(body).toHaveProperty('profile');
    expect(['local', 'single', 'airgap']).toContain(body.profile);
    expect(body).toHaveProperty('model');
    expect(body).toHaveProperty('feature_flags');
    expect(body.feature_flags).toHaveProperty('use_rag_pipeline');
    expect(body.feature_flags).toHaveProperty('model_version');
    expect(body).toHaveProperty('agents_loaded');
    expect(body).toHaveProperty('indexes');
    expect(body).toHaveProperty('bm25_indexes');
    expect(body).toHaveProperty('session_store');
    expect(body.session_store).toHaveProperty('driver', 'sqlite');
  });

  test('OpenAPI advertises shell-first runtime endpoints', async ({ request }) => {
    const response = await request.get('/openapi.json');

    expect(response.status()).toBe(200);

    const schema = await response.json();
    expect(schema).toHaveProperty('paths');

    for (const path of [
      '/health',
      '/search',
      '/v1/search',
      '/v1/generate',
      '/v1/generate-civil-response',
      '/v1/stream',
      '/v1/agent/run',
      '/v1/agent/stream',
    ]) {
      expect(schema.paths).toHaveProperty(path);
    }

    expect(schema.paths).not.toHaveProperty('/v1/classify');
  });
});

test.describe('GovOn Runtime Security Contract', () => {
  test('protected runtime endpoints reject missing API keys', async ({ request }) => {
    const response = await request.post('/v1/search', {
      data: searchPayload,
      failOnStatusCode: false,
    });

    expect(response.status()).toBe(401);

    const body = await response.json();
    expect(body).toHaveProperty('detail');
  });

  test('protected runtime endpoints accept API key and degrade gracefully in SKIP_MODEL_LOAD mode', async ({
    request,
  }) => {
    const response = await request.post('/v1/search', {
      data: searchPayload,
      headers: {
        'X-API-Key': apiKey,
      },
      failOnStatusCode: false,
    });

    expect(response.status()).toBe(503);

    const body = await response.json();
    expect(body).toHaveProperty('detail', '검색 엔진이 아직 초기화되지 않았습니다.');
  });

  test('unknown paths still return 404', async ({ request }) => {
    const response = await request.get('/nonexistent-path', {
      failOnStatusCode: false,
    });

    expect(response.status()).toBe(404);
  });
});
