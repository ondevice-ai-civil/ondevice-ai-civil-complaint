/**
 * ApprovalPrompt — renders a human-approval gate dialog.
 *
 * Shown when the backend emits an `awaiting_approval` event.
 * The user presses 'y' to approve or 'n' to reject.
 * Calls onApprove(true) or onApprove(false) accordingly.
 */

import React, { useCallback } from 'react';
import { Box, Text, useInput } from 'ink';
import type { ApprovalRequest } from '../types.js';
import { TASK_TYPE_LABELS, TASK_TYPE_STYLES } from '../types.js';
import { THEME_COLORS } from '../config.js';

interface ApprovalPromptProps {
  request: ApprovalRequest;
  onApprove: (approved: boolean) => void;
}

export function ApprovalPrompt({ request, onApprove }: ApprovalPromptProps) {
  useInput(
    useCallback(
      (input: string) => {
        const key = input.toLowerCase();
        if (key === 'y') {
          onApprove(true);
        } else if (key === 'n') {
          onApprove(false);
        }
      },
      [onApprove],
    ),
  );

  const taskLabel =
    TASK_TYPE_LABELS[request.task_type] ?? TASK_TYPE_LABELS['default'];
  const taskColor =
    TASK_TYPE_STYLES[request.task_type] ?? TASK_TYPE_STYLES['default'];

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={THEME_COLORS.warning}
      paddingX={2}
      paddingY={1}
      marginTop={1}
    >
      {/* Header */}
      <Box flexDirection="row">
        <Text bold color={THEME_COLORS.warning}>{'⚠ 승인 요청  '}</Text>
        <Text color={taskColor}>[{taskLabel}]</Text>
      </Box>

      {/* Description */}
      {request.description && (
        <Box marginTop={1}>
          <Text wrap="wrap">{request.description}</Text>
        </Box>
      )}

      {/* Planned tools list */}
      {request.planned_tools && request.planned_tools.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text color={THEME_COLORS.muted}>{'실행 예정 도구:'}</Text>
          {request.planned_tools.map((tool, i) => (
            <Text key={i} color={THEME_COLORS.muted}>{`  • ${tool}`}</Text>
          ))}
        </Box>
      )}

      {/* Prompt */}
      <Box marginTop={1}>
        <Text bold>
          {'승인하시겠습니까? '}
          <Text color="green">{'y'}</Text>
          {' / '}
          <Text color="red">{'n'}</Text>
        </Text>
      </Box>
    </Box>
  );
}
