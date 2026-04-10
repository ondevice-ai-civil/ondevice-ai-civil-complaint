/**
 * ApprovalPrompt — renders a human-approval gate dialog.
 *
 * Shown when the backend emits an `awaiting_approval` event.
 * Uses @inkjs/ui Select for keyboard-navigable approve/reject choice.
 * Calls onApprove(true) or onApprove(false) on selection.
 *
 * Design: B-4 — Claude Code style box-title + @inkjs/ui Select navigation.
 */

import React, { useRef } from 'react';
import { Box, Text } from 'ink';
import { Select } from '@inkjs/ui';
import type { ApprovalRequest } from '../types.js';
import { TASK_TYPE_LABELS, TASK_TYPE_STYLES } from '../types.js';
import { THEME_COLORS } from '../config.js';

interface ApprovalPromptProps {
  request: ApprovalRequest;
  onApprove: (approved: boolean) => void;
}

const MAX_DESCRIPTION_LENGTH = 1000;

const SELECT_OPTIONS = [
  { label: '✓ 승인', value: 'approve' },
  { label: '✗ 거부', value: 'reject' },
];

export function ApprovalPrompt({ request, onApprove }: ApprovalPromptProps) {
  const safeDescription = request.description?.slice(0, MAX_DESCRIPTION_LENGTH);
  // Prevent duplicate submissions if onChange fires more than once
  const decidedRef = useRef(false);

  const handleChange = (value: string) => {
    if (decidedRef.current) return;
    decidedRef.current = true;
    onApprove(value === 'approve');
  };

  const taskLabel =
    TASK_TYPE_LABELS[request.task_type] ?? TASK_TYPE_LABELS['default'];
  const taskColor =
    TASK_TYPE_STYLES[request.task_type] ?? TASK_TYPE_STYLES['default'];

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={THEME_COLORS.primary}
      paddingX={2}
      paddingY={1}
      marginTop={1}
    >
      {/* Box title row */}
      <Box flexDirection="row" marginBottom={1}>
        <Text bold color={THEME_COLORS.primary}>{'승인 요청  '}</Text>
        <Text color={taskColor}>[{taskLabel}]</Text>
      </Box>

      {/* Description body */}
      {safeDescription && (
        <Box marginBottom={1}>
          <Text wrap="wrap">{safeDescription}</Text>
        </Box>
      )}

      {/* Planned tools list */}
      {request.planned_tools && request.planned_tools.length > 0 && (
        <Box flexDirection="column" marginBottom={1}>
          <Text color={THEME_COLORS.muted}>{'실행 예정 도구:'}</Text>
          {request.planned_tools.map((tool, i) => (
            <Text key={i} color={THEME_COLORS.muted}>{`  • ${tool}`}</Text>
          ))}
        </Box>
      )}

      {/* Select navigation */}
      <Box marginTop={1}>
        <Select options={SELECT_OPTIONS} onChange={handleChange} />
      </Box>
    </Box>
  );
}
