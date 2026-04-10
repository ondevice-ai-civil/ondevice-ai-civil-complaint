import { useState, useEffect, useRef } from 'react';
import type { GovOnClient } from '../client.js';

interface DaemonState {
  /** Server is confirmed ready. */
  ready: boolean;
  /** Currently waiting for server. */
  waiting: boolean;
  /** Error message if server could not be reached. */
  error: string | null;
}

/**
 * Check server health on mount. If the server is sleeping (e.g. HF Space),
 * call client.waitForReady() which polls with status messages to stderr.
 */
export function useDaemon(client: GovOnClient): DaemonState {
  const [state, setState] = useState<DaemonState>({
    ready: false,
    waiting: true,
    error: null,
  });
  const controllerRef = useRef(new AbortController());

  useEffect(() => {
    let mounted = true;

    async function check() {
      try {
        // Quick health check first
        await client.health();
        if (mounted) setState({ ready: true, waiting: false, error: null });
      } catch {
        // Server not immediately available — try cold start wait
        if (mounted) setState((s) => ({ ...s, waiting: true }));
        try {
          const ok = await client.waitForReady();
          if (mounted) {
            setState({
              ready: ok,
              waiting: false,
              error: ok ? null : 'Server connection timed out',
            });
          }
        } catch (err) {
          if (mounted) {
            setState({
              ready: false,
              waiting: false,
              error: err instanceof Error ? err.message : String(err),
            });
          }
        }
      }
    }

    check();
    return () => {
      mounted = false;
      controllerRef.current.abort();
    };
  }, [client]);

  return state;
}
