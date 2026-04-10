import React from 'react';
import { Box, Text } from 'ink';
import { Spinner } from '@inkjs/ui';
import { THEME_COLORS } from '../config.js';
import type { ToolInvocation } from '../types.js';

interface ToolPanelProps {
  tools: ToolInvocation[];
}

export function ToolPanel({ tools }: ToolPanelProps) {
  if (tools.length === 0) return null;

  const total = tools.length;
  const pendingCount = tools.filter((t) => t.pending).length;
  const failedCount = tools.filter((t) => !t.pending && t.success === false).length;
  const doneCount = total - pendingCount;

  // While any tool is still running, show a single spinner line
  if (pendingCount > 0) {
    return (
      <Box marginLeft={2} gap={1}>
        <Spinner />
        <Text color={THEME_COLORS.muted}>{total}개 도구 실행 중…</Text>
      </Box>
    );
  }

  // All done — show success-only or mixed summary
  if (failedCount === 0) {
    return (
      <Box marginLeft={2}>
        <Text color={THEME_COLORS.success}>✓ {doneCount}개 도구 실행 완료</Text>
      </Box>
    );
  }

  const succeededCount = doneCount - failedCount;
  return (
    <Box marginLeft={2} gap={1}>
      <Text color={THEME_COLORS.success}>✓ {succeededCount}개 완료</Text>
      <Text color={THEME_COLORS.muted}>·</Text>
      <Text color={THEME_COLORS.error}>✗ {failedCount}개 실패</Text>
    </Box>
  );
}
