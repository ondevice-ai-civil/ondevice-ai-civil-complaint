import React, { useEffect, useState } from 'react';
import { Box, Text } from 'ink';
import { Spinner } from '@inkjs/ui';
import { THEME_COLORS } from '../config.js';

interface ThinkingBlockProps {
  /** Accumulated thinking text (may be streaming). */
  content: string;
  /** Whether thinking is still in progress. */
  streaming?: boolean;
}

export function ThinkingBlock({ content, streaming = false }: ThinkingBlockProps) {
  // Elapsed time in milliseconds, updated every second while streaming
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

  // Completed state: show elapsed time captured at completion
  const elapsedSeconds = elapsedMs / 1000;
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
