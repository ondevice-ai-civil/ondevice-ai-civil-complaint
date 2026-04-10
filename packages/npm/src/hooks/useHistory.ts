import { useReducer, useCallback, useEffect, useRef } from 'react';
import { writeFile } from 'node:fs/promises';
import { readFileSync, mkdirSync, lstatSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const HISTORY_DIR = join(homedir(), '.govon');
const HISTORY_FILE = join(HISTORY_DIR, 'history');
const MAX_ENTRIES = 500;
const MAX_ENTRY_BYTES = 4096;
const DEBOUNCE_MS = 300;

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

interface HistoryState {
  entries: string[];
  cursor: number; // -1 = at current input
}

type HistoryAction =
  | { type: 'PUSH'; entry: string }
  | { type: 'NAVIGATE'; direction: 'up' | 'down' }
  | { type: 'RESET_CURSOR' };

// ---------------------------------------------------------------------------
// Reducer — makes push (entries + cursor) atomic
// ---------------------------------------------------------------------------

function historyReducer(state: HistoryState, action: HistoryAction): HistoryState {
  switch (action.type) {
    case 'PUSH': {
      const trimmed = action.entry.trim();
      if (!trimmed) return state;
      if (Buffer.byteLength(trimmed, 'utf8') > MAX_ENTRY_BYTES) return state;

      // Deduplicate consecutive identical entries
      const next =
        state.entries[state.entries.length - 1] === trimmed
          ? state.entries
          : [...state.entries, trimmed];

      return { entries: next.slice(-MAX_ENTRIES), cursor: -1 };
    }

    case 'NAVIGATE': {
      const { entries, cursor } = state;
      if (entries.length === 0) return state;

      if (action.direction === 'up') {
        const next = cursor === -1 ? entries.length - 1 : Math.max(0, cursor - 1);
        return { ...state, cursor: next };
      }

      // direction === 'down'
      if (cursor === -1) return state; // already at current input — nothing to do
      if (cursor + 1 >= entries.length) {
        return { ...state, cursor: -1 }; // back to empty input
      }
      return { ...state, cursor: cursor + 1 };
    }

    case 'RESET_CURSOR':
      return { ...state, cursor: -1 };

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Initial state loader
// ---------------------------------------------------------------------------

function loadInitialEntries(): string[] {
  try {
    const raw = readFileSync(HISTORY_FILE, 'utf-8');
    return raw.split('\n').filter(Boolean).slice(-MAX_ENTRIES);
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Async persist helper — symlink-safe
// ---------------------------------------------------------------------------

async function persistEntries(entries: string[]): Promise<void> {
  try {
    mkdirSync(HISTORY_DIR, { recursive: true });

    // Reject symlinks to prevent symlink-based write attacks
    try {
      const stat = lstatSync(HISTORY_FILE);
      if (!stat.isFile()) return; // reject symlinks or other non-regular files
    } catch {
      // File does not exist yet — safe to create
    }

    await writeFile(HISTORY_FILE, entries.join('\n') + '\n', 'utf-8');
  } catch {
    // Silently ignore write failures (read-only fs, etc.)
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Load and persist CLI input history from ~/.govon/history.
 * Returns the history entries, a push function, a navigate function, and the
 * current cursor position (-1 means "at current input").
 */
export function useHistory() {
  const [state, dispatch] = useReducer(historyReducer, undefined, () => ({
    entries: loadInitialEntries(),
    cursor: -1,
  }));

  const { entries, cursor } = state;

  // Debounce + overlap-guard refs for async persistence
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const writeInProgressRef = useRef(false);
  const pendingWriteRef = useRef(false);
  const latestEntriesRef = useRef(entries);

  // Keep latest entries ref in sync
  latestEntriesRef.current = entries;

  // Persist entries whenever they change, with debounce and trailing-write guard
  useEffect(() => {
    if (debounceTimerRef.current !== null) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null;

      if (writeInProgressRef.current) {
        // A write is running — queue a trailing write so the latest state is persisted
        pendingWriteRef.current = true;
        return;
      }

      writeInProgressRef.current = true;
      persistEntries(latestEntriesRef.current).finally(() => {
        writeInProgressRef.current = false;
        // If entries changed while the write was in progress, flush the latest
        if (pendingWriteRef.current) {
          pendingWriteRef.current = false;
          writeInProgressRef.current = true;
          void persistEntries(latestEntriesRef.current).finally(() => {
            writeInProgressRef.current = false;
          });
        }
      });
    }, DEBOUNCE_MS);

    return () => {
      if (debounceTimerRef.current !== null) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }
    };
  }, [entries]);

  const push = useCallback((entry: string) => {
    dispatch({ type: 'PUSH', entry });
  }, []);

  // Navigate up/down through history.
  // Returns the selected entry string, '' when returning to current input, or
  // undefined when there is nothing to navigate.
  const navigate = useCallback(
    (direction: 'up' | 'down'): string | undefined => {
      if (entries.length === 0) return undefined;

      const prevCursor = cursor;

      if (direction === 'down') {
        if (prevCursor === -1) return undefined; // already at current input
        if (prevCursor + 1 >= entries.length) {
          dispatch({ type: 'NAVIGATE', direction: 'down' });
          return ''; // signal: restore empty input
        }
      }

      dispatch({ type: 'NAVIGATE', direction });

      // Compute the next cursor locally so we can return the correct entry
      // without waiting for the next render cycle.
      if (direction === 'up') {
        const next = prevCursor === -1 ? entries.length - 1 : Math.max(0, prevCursor - 1);
        return entries[next];
      } else {
        return entries[prevCursor + 1];
      }
    },
    [entries, cursor],
  );

  const resetCursor = useCallback(() => {
    dispatch({ type: 'RESET_CURSOR' });
  }, []);

  return { entries, push, navigate, resetCursor, cursor };
}
