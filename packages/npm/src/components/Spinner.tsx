import React, { useState, useEffect } from 'react';
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

/** Pick a random Korean proverb from the verb list. */
function randomVerb(): string {
  return SPINNER_VERBS[Math.floor(Math.random() * SPINNER_VERBS.length)];
}

export function Spinner({ dotColor = THEME_COLORS.warning, tokens = 0 }: SpinnerProps) {
  const [elapsed, setElapsed] = useState(0);
  const [verb, setVerb] = useState(randomVerb);

  // Increment elapsed time every second
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsed((s) => s + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Rotate proverb every ~3 seconds
  useEffect(() => {
    const rotator = setInterval(() => {
      setVerb(randomVerb());
    }, 3000);
    return () => clearInterval(rotator);
  }, []);

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
