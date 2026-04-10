import { useState, useCallback, useEffect } from 'react';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const HISTORY_DIR = join(homedir(), '.govon');
const HISTORY_FILE = join(HISTORY_DIR, 'history');
const MAX_ENTRIES = 500;

/**
 * Load and persist CLI input history from ~/.govon/history.
 * Returns the history array, a push function, and a navigate function.
 */
export function useHistory() {
  const [entries, setEntries] = useState<string[]>(() => {
    try {
      const raw = readFileSync(HISTORY_FILE, 'utf-8');
      return raw.split('\n').filter(Boolean).slice(-MAX_ENTRIES);
    } catch {
      return [];
    }
  });

  const [cursor, setCursor] = useState(-1);

  const push = useCallback((entry: string) => {
    const trimmed = entry.trim();
    if (!trimmed) return;
    setEntries((prev) => {
      // Deduplicate consecutive
      const next = prev[prev.length - 1] === trimmed ? prev : [...prev, trimmed];
      return next.slice(-MAX_ENTRIES);
    });
    setCursor(-1);
  }, []);

  // Navigate up/down through history
  const navigate = useCallback(
    (direction: 'up' | 'down'): string | undefined => {
      if (entries.length === 0) return undefined;

      let next: number;
      if (direction === 'up') {
        next = cursor === -1 ? entries.length - 1 : Math.max(0, cursor - 1);
      } else {
        next = cursor === -1 ? -1 : Math.min(entries.length - 1, cursor + 1);
        if (next >= entries.length) {
          setCursor(-1);
          return '';
        }
      }
      setCursor(next);
      return entries[next];
    },
    [entries, cursor],
  );

  // Persist on change
  useEffect(() => {
    try {
      mkdirSync(HISTORY_DIR, { recursive: true });
      writeFileSync(HISTORY_FILE, entries.join('\n') + '\n', 'utf-8');
    } catch {
      // Silently ignore write failures (read-only fs, etc.)
    }
  }, [entries]);

  return { entries, push, navigate, cursor };
}
