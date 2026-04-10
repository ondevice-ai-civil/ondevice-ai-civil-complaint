import React, { useState, useEffect, useRef } from 'react';
import { Box, Text } from 'ink';
import { Spinner as InkSpinner } from '@inkjs/ui';
import { THEME_COLORS } from '../config.js';
import { SPINNER_VERBS } from '../data/spinnerVerbs.js';

interface SpinnerProps {
  /** Color for the status dot. */
  dotColor?: string;
  /** Token count to display. */
  tokens?: number;
}

/** Format elapsed seconds into a human-readable duration string. */
function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

export function Spinner({ dotColor = THEME_COLORS.warning, tokens = 0 }: SpinnerProps) {
  const [elapsed, setElapsed] = useState(0);
  // Freeze the random starting verb index so it does not shift on every render
  const startVerbIndexRef = useRef(Math.floor(Math.random() * SPINNER_VERBS.length));

  // Single interval drives both elapsed counter and proverb rotation.
  // Proverb index is derived from elapsed ticks so no second timer is needed.
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed((s) => s + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Rotate verb every 3 ticks (3 seconds) deterministically
  const verb = SPINNER_VERBS[(startVerbIndexRef.current + Math.floor(elapsed / 3)) % SPINNER_VERBS.length];
  const elapsedStr = formatElapsed(elapsed);

  return (
    <Box flexDirection="row" gap={1}>
      <Text color={dotColor}>●</Text>
      <Text bold color={THEME_COLORS.accent}>
        GovOn
      </Text>
      <InkSpinner />
      <Text dimColor>
        {verb}… ({elapsedStr})
      </Text>
    </Box>
  );
}
