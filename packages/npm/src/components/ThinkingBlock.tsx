import React from 'react';
import { Box, Text } from 'ink';
import { THEME_COLORS } from '../config.js';

interface ThinkingBlockProps {
  /** Accumulated thinking text (may be streaming). */
  content: string;
  /** Whether thinking is still in progress. */
  streaming?: boolean;
}

export function ThinkingBlock({ content, streaming = false }: ThinkingBlockProps) {
  if (!content) return null;

  return (
    <Box flexDirection="column" marginLeft={2}>
      <Text color={THEME_COLORS.muted} dimColor>
        {'💭 '}{streaming ? '사고 중…' : '사고 완료'}
      </Text>
      <Box marginLeft={2}>
        <Text color={THEME_COLORS.dimmed} dimColor wrap="wrap">
          {content}
        </Text>
      </Box>
    </Box>
  );
}
