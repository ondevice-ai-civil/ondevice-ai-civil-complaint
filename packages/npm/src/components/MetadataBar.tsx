import React from 'react';
import { Box, Text } from 'ink';
import { THEME_COLORS } from '../config.js';
import type { RunMetadata } from '../types.js';

interface MetadataBarProps {
  metadata: RunMetadata;
}

export function MetadataBar({ metadata }: MetadataBarProps) {
  const parts: string[] = [];
  if (metadata.total_iterations !== undefined) {
    parts.push(`${metadata.total_iterations} iterations`);
  }
  if (metadata.total_tool_calls !== undefined) {
    parts.push(`${metadata.total_tool_calls} tools`);
  }
  if (metadata.total_latency_ms !== undefined) {
    parts.push(`${Math.round(metadata.total_latency_ms).toLocaleString()}ms`);
  }

  if (parts.length === 0) return null;

  return (
    <Box marginLeft={2} marginTop={1}>
      <Text color={THEME_COLORS.muted} dimColor>
        ─ {parts.join(' · ')}
      </Text>
    </Box>
  );
}
