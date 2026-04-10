/**
 * Client factory — returns MockGovOnClient or GovOnClient based on GOVON_MOCK env var.
 *
 * Usage:  const client = createClient();
 */

import type { IClient } from './client.interface.js';
import { GovOnClient } from './client.js';
import { MockGovOnClient } from './mockClient.js';
import { getBaseUrl } from './config.js';

/** true when GOVON_MOCK is set to any truthy value. */
export const isMockMode = ['true', '1', 'yes'].includes(
  (process.env.GOVON_MOCK ?? '').toLowerCase(),
);

/**
 * Create the appropriate client for the current environment.
 *
 * @param baseUrl - Only used for the real client; ignored in mock mode.
 */
export function createClient(baseUrl?: string): IClient {
  if (isMockMode) {
    return new MockGovOnClient();
  }
  return new GovOnClient(baseUrl ?? getBaseUrl());
}
