/**
 * MessageBubble — renders a single completed chat message (user or assistant).
 *
 * For user messages: plain text with a ❯ prefix.
 * For assistant messages: markdown-rendered content, optional thinking steps,
 * optional tool invocations, optional evidence/metadata footers.
 */

import React from 'react';
import { Box, Text } from 'ink';
import type { Message } from '../types.js';
import { THEME_COLORS } from '../config.js';
import { MarkdownView } from './MarkdownView.js';
import { ThinkingBlock } from './ThinkingBlock.js';
import { ToolPanel } from './ToolPanel.js';
import { MetadataBar } from './MetadataBar.js';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === 'user') {
    return (
      <Box flexDirection="row" marginTop={1}>
        <Text bold color={THEME_COLORS.accent}>{'❯ '}</Text>
        <Text backgroundColor="#2a2a2a" color={THEME_COLORS.accent}>{' ' + message.content + ' '}</Text>
      </Box>
    );
  }

  // Status dot color: error > streaming > success
  const dotColor = message.error
    ? THEME_COLORS.error
    : message.streaming
      ? THEME_COLORS.warning
      : THEME_COLORS.success;

  // Assistant message
  return (
    <Box flexDirection="column" marginTop={1}>
      {/* Role label: status dot + brand name */}
      <Box flexDirection="row">
        <Text color={dotColor}>{'● '}</Text>
        <Text bold color={THEME_COLORS.accent}>{'GovOn'}</Text>
      </Box>

      {/* Thinking steps (collapsed after completion) */}
      {message.thinking && message.thinking.length > 0 && (
        <Box flexDirection="column">
          {message.thinking.map((step, i) => (
            <ThinkingBlock
              key={i}
              content={step.content}
              streaming={false}
            />
          ))}
        </Box>
      )}

      {/* Tool invocations */}
      {message.tools && message.tools.length > 0 && (
        <ToolPanel tools={message.tools} />
      )}

      {/* Main answer content */}
      {message.error ? (
        <Box marginLeft={2}>
          <Text color={THEME_COLORS.error}>{'오류: '}{message.error}</Text>
        </Box>
      ) : (
        <Box marginLeft={2}>
          <MarkdownView content={message.content} streaming={false} />
        </Box>
      )}

      {/* Run metadata footer (evidence display is handled elsewhere) */}
      {message.metadata && (
        <MetadataBar metadata={message.metadata} />
      )}
    </Box>
  );
}
