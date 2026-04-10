import React from 'react';
import { Box, Text } from 'ink';
import { THEME_COLORS } from '../config.js';
import type { RunMetadata } from '../types.js';

interface MetadataBarProps {
  metadata: RunMetadata;
}

export function MetadataBar({ metadata }: MetadataBarProps) {
  const segments: string[] = [];

  // Thinking iterations: ↻N
  if (metadata.total_iterations !== undefined) {
    segments.push(`↻${metadata.total_iterations}`);
  }

  // Tool calls: ⚙N
  if (metadata.total_tool_calls !== undefined) {
    segments.push(`⚙${metadata.total_tool_calls}`);
  }

  // Latency: ⏱ X.Xs
  if (metadata.total_latency_ms !== undefined) {
    const seconds = (metadata.total_latency_ms / 1000).toFixed(1);
    segments.push(`⏱ ${seconds}s`);
  }

  if (segments.length === 0) return null;

  return (
    <Box justifyContent="flex-end" marginTop={1}>
      <Text color={THEME_COLORS.dimmed} dimColor>
        {segments.join(' ')}
      </Text>
    </Box>
  );
}
