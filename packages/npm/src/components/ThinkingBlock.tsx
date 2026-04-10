import React, { useEffect, useState } from 'react';
import { Box, Text } from 'ink';
import { Spinner } from '@inkjs/ui';
import { THEME_COLORS } from '../config.js';

interface ThinkingBlockProps {
  /** Accumulated thinking text (may be streaming). */
  content: string;
  /** Whether thinking is still in progress. */
  streaming?: boolean;
  /**
   * Externally measured elapsed duration in milliseconds.
   * When provided and streaming=false, this value is used instead of the
   * internal timer (which would show 0s for messages loaded from history).
   */
  elapsedMs?: number;
}

export function ThinkingBlock({ content, streaming = false, elapsedMs: externalElapsedMs }: ThinkingBlockProps) {
  // Internal elapsed time in milliseconds, updated every second while streaming
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (!streaming) return;

    // Reset elapsed time when streaming starts
    setElapsedMs(0);
    const startTime = Date.now();

    const interval = setInterval(() => {
      setElapsedMs(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [streaming]);

  if (!content) return null;

  if (streaming) {
    const elapsedSeconds = Math.floor(elapsedMs / 1000);
    return (
      <Box flexDirection="row" marginLeft={2}>
        <Spinner />
        <Text color={THEME_COLORS.muted}>{' 사고 중… ('}{elapsedSeconds}{'s)'}</Text>
      </Box>
    );
  }

  // Completed state: prefer externally measured elapsed over internal timer
  // (internal timer shows 0s for pre-rendered historical messages)
  const resolvedElapsedMs = externalElapsedMs ?? elapsedMs;
  const elapsedSeconds = resolvedElapsedMs / 1000;
  const elapsedDisplay = elapsedSeconds % 1 === 0
    ? `${elapsedSeconds.toFixed(0)}s`
    : `${elapsedSeconds.toFixed(1)}s`;

  return (
    <Box flexDirection="row" marginLeft={2}>
      <Text color={THEME_COLORS.muted} dimColor>
        {'⏵ 사고 완료 ('}{elapsedDisplay}{')'}
      </Text>
    </Box>
  );
}
