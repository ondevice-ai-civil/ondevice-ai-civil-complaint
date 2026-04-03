import { expect, test, type APIRequestContext } from '@playwright/test';

type OpenApiSchema = {
  components?: {
    schemas?: Record<string, unknown>;
  };
  paths: Record<string, unknown>;
};

async function getOpenApiSchema(request: APIRequestContext): Promise<OpenApiSchema> {
  const response = await request.get('/openapi.json');
  expect(response.status()).toBe(200);
  return response.json();
}

function resolveSchemaRef(schema: OpenApiSchema, maybeRef: unknown): Record<string, unknown> | undefined {
  if (!maybeRef || typeof maybeRef !== 'object') {
    return undefined;
  }

  if ('$ref' in maybeRef && typeof maybeRef.$ref === 'string') {
    const parts = maybeRef.$ref.split('/');
    const schemaName = parts[parts.length - 1];
    const resolved = schema.components?.schemas?.[schemaName];
    if (resolved && typeof resolved === 'object') {
      return resolved as Record<string, unknown>;
    }
    return undefined;
  }

  return maybeRef as Record<string, unknown>;
}

test.describe('Session Runtime Forward Contract', () => {
  test('/api/v2/sessions exposes a session bootstrap contract once implemented', async ({
    request,
  }) => {
    const schema = await getOpenApiSchema(request);
    const pathItem = schema.paths['/api/v2/sessions'] as Record<string, unknown> | undefined;

    test.skip(!pathItem, 'Session runtime endpoints are not available on this branch yet.');

    const post = pathItem?.post as Record<string, unknown> | undefined;
    const requestSchema = resolveSchemaRef(
      schema,
      (post?.requestBody as Record<string, any>)?.content?.['application/json']?.schema,
    );

    expect(requestSchema).toBeDefined();
    expect(requestSchema).toHaveProperty('properties');

    const properties = requestSchema?.properties as Record<string, unknown>;
    expect(properties).toHaveProperty('officer_id');
    expect(properties).toHaveProperty('title');
  });

  test('/api/v2/messages and system status ship together with session runtime', async ({
    request,
  }) => {
    const schema = await getOpenApiSchema(request);
    const hasSessions = Boolean(schema.paths['/api/v2/sessions']);
    const hasMessages = Boolean(schema.paths['/api/v2/sessions/{id}/messages']);
    const hasSystemStatus = Boolean(schema.paths['/api/v2/system/status']);

    test.skip(!hasSessions, 'Session runtime endpoints are not available on this branch yet.');
    expect(hasMessages).toBeTruthy();
    expect(hasSystemStatus).toBeTruthy();
  });

  test('/api/v2/sessions returns session metadata once implemented', async ({ request }) => {
    const schema = await getOpenApiSchema(request);
    const pathItem = schema.paths['/api/v2/sessions'] as Record<string, unknown> | undefined;

    test.skip(!pathItem, 'Session runtime endpoints are not available on this branch yet.');

    const post = pathItem?.post as Record<string, unknown> | undefined;
    const responseSchema = resolveSchemaRef(
      schema,
      (post?.responses as Record<string, any>)?.['201']?.content?.['application/json']?.schema,
    );

    expect(responseSchema).toBeDefined();
    expect(responseSchema).toHaveProperty('properties');

    const properties = responseSchema?.properties as Record<string, unknown>;
    expect(properties).toHaveProperty('id');
    expect(properties).toHaveProperty('officer_id');
    expect(properties).toHaveProperty('status');
    expect(properties).toHaveProperty('total_turns');
    expect(properties).toHaveProperty('websocket_url');
  });
});
