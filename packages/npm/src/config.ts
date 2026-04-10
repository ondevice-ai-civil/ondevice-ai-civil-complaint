/**
 * Runtime configuration loaded from environment variables.
 * All timeouts are in milliseconds.
 */

/** Parse a positive integer from env, clamped to [1, maxVal]. Falls back to defaultVal. */
function parsePosInt(envVal: string | undefined, defaultVal: number, maxVal = 86_400): number {
  const parsed = parseInt(envVal ?? String(defaultVal), 10);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > maxVal) {
    return defaultVal;
  }
  return parsed;
}

/** Resolve and validate the backend base URL from GOVON_RUNTIME_URL. */
export function getBaseUrl(): string {
  const raw = (process.env.GOVON_RUNTIME_URL ?? 'http://127.0.0.1:8000').replace(/\/+$/, '');
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      throw new Error(`GOVON_RUNTIME_URL must use http or https scheme, got: ${parsed.protocol}`);
    }
    // Warn when sending plaintext to a non-localhost remote
    if (
      parsed.hostname !== '127.0.0.1' &&
      parsed.hostname !== 'localhost' &&
      parsed.protocol === 'http:'
    ) {
      process.stderr.write(
        '[warn] GOVON_RUNTIME_URL uses http:// for a non-localhost host. ' +
          'Use https:// to protect query content in transit.\n',
      );
    }
    return raw;
  } catch (err) {
    if (err instanceof Error && err.message.startsWith('GOVON_RUNTIME_URL')) throw err;
    throw new Error(`GOVON_RUNTIME_URL is not a valid URL: ${raw}`);
  }
}

// Timeout configuration in milliseconds
export const TIMEOUTS = {
  connect: 10_000,
  read: 300_000,
  /** Blocking /v2/agent/run and /v3/agent/run request timeout. */
  run: 120_000,
  /** Default timeout for simple requests (health, approve, cancel). */
  default: 30_000,
  coldStart: parsePosInt(process.env.GOVON_COLD_START_TIMEOUT, 600) * 1000,
  coldStartInterval: parsePosInt(process.env.GOVON_COLD_START_INTERVAL, 5) * 1000,
} as const;

// Brand color palette
export const THEME_COLORS = {
  primary: '#2D5A3D',
  accent: '#F5E6C8',
  muted: '#7A8B7E',
  error: '#D32F2F',
  warning: '#FFA000',
  success: '#388E3C',
  info: '#1976D2',
  dimmed: '#666666',
} as const;

// Spinner animation config
export const SPINNER = {
  fps: 30,
  verbChangeInterval: 90, // frames (~3 seconds at 30fps)
  chars: ['\u00b7', '\u273b', '\u273d', '\u2736', '\u2733', '\u2722'] as const,
} as const;
