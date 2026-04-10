import React from 'react';
import { Box, Text } from 'ink';
import { THEME_COLORS } from '../config.js';
import { TOOL_DISPLAY_NAMES } from '../types.js';
import type { ToolInvocation } from '../types.js';

interface ToolPanelProps {
  tools: ToolInvocation[];
}

export function ToolPanel({ tools }: ToolPanelProps) {
  if (tools.length === 0) return null;

  return (
    <Box flexDirection="column" marginLeft={2}>
      {tools.map((tool, i) => {
        const displayName = TOOL_DISPLAY_NAMES[tool.tool] ?? tool.tool;
        if (tool.pending) {
          return (
            <Text key={i} color={THEME_COLORS.warning}>
              ┌─ ⚙ {tool.tool} ({displayName}) 실행 중…
            </Text>
          );
        }
        const statusColor = tool.success !== false ? THEME_COLORS.success : THEME_COLORS.error;
        const statusIcon = tool.success !== false ? '✦' : '✘';
        const statusText = tool.success !== false ? '완료' : '실패';
        return (
          <Text key={i} color={statusColor}>
            └─ {statusIcon} {tool.tool} ({displayName}) {statusText}
          </Text>
        );
      })}
    </Box>
  );
}
