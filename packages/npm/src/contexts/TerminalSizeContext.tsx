/**
 * TerminalSizeContext.tsx — shared terminal dimension context.
 *
 * Provides a single resize subscription for the entire app.
 * Components call useTerminalSize() instead of each managing their own
 * stdout "resize" listener via useStdout().
 */

import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { useStdout } from 'ink';

interface TerminalSize {
  columns: number;
  rows: number;
}

const DEFAULT_SIZE: TerminalSize = { columns: 80, rows: 24 };

const TerminalSizeContext = createContext<TerminalSize>(DEFAULT_SIZE);

/** Wraps children with a shared terminal size subscription. */
export function TerminalSizeProvider({ children }: { children: React.ReactNode }) {
  const { stdout } = useStdout();
  const [size, setSize] = useState<TerminalSize>({
    columns: stdout?.columns ?? 80,
    rows: stdout?.rows ?? 24,
  });

  useEffect(() => {
    if (!stdout) return;
    const onResize = () => {
      setSize({ columns: stdout.columns ?? 80, rows: stdout.rows ?? 24 });
    };
    stdout.on('resize', onResize);
    return () => {
      stdout.off('resize', onResize);
    };
  }, [stdout]);

  // Memoize by value so downstream components only re-render when dimensions change
  const value = useMemo(() => size, [size.columns, size.rows]);

  return (
    <TerminalSizeContext.Provider value={value}>
      {children}
    </TerminalSizeContext.Provider>
  );
}

/** Returns the current terminal dimensions from the nearest TerminalSizeProvider. */
export function useTerminalSize(): TerminalSize {
  return useContext(TerminalSizeContext);
}
