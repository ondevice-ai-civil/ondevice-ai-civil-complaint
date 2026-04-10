// Environment variable for server URL
// When GOVON_RUNTIME_URL is set, connect to remote server directly (skip local daemon)
export function getBaseUrl(): string {
  return (process.env.GOVON_RUNTIME_URL ?? 'http://127.0.0.1:8000').replace(/\/+$/, '');
}

// Timeout configuration in milliseconds
export const TIMEOUTS = {
  connect: 10_000,
  read: 300_000,
  run: 120_000,
  default: 30_000,
  coldStart: parseInt(process.env.GOVON_COLD_START_TIMEOUT ?? '600', 10) * 1000,
  coldStartInterval: parseInt(process.env.GOVON_COLD_START_INTERVAL ?? '5', 10) * 1000,
} as const;

// Brand color palette
export const THEME = {
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
