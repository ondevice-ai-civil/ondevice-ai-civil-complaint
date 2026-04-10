/**
 * MessageList — viewport-based message list for the alternate screen buffer.
 *
 * Replaces Ink's <Static> component. Renders Banner + all messages within a
 * fixed-height Box so that content stays within the alternate screen (no
 * scrollback leakage). Supports manual scrolling via Shift+Up/Down and
 * PageUp/PageDown keys.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import type { Message } from '../types.js';
import { Banner } from './Banner.js';
import { MessageBubble } from './MessageBubble.js';

interface MessageListProps {
  messages: Message[];
  version: string;
  /** Available height in terminal rows for the message area. */
  height: number;
}

export function MessageList({ messages, version, height }: MessageListProps) {
  const [scrollOffset, setScrollOffset] = useState(0);
  const [autoScroll, setAutoScroll] = useState(true);

  // Reset to bottom when new messages arrive (if auto-scroll is on).
  useEffect(() => {
    if (autoScroll) {
      setScrollOffset(0);
    }
  }, [messages.length, autoScroll]);

  // Max scroll: always keep at least 1 message visible
  const maxScroll = Math.max(0, messages.length - 1);

  // Scroll key handling: Shift+Up/Down (1 message), PageUp/PageDown (10 messages),
  // Meta+Up (Home / scroll to top), Meta+Down (End / scroll to bottom).
  useInput((_input, key) => {
    if (key.shift && key.upArrow) {
      setScrollOffset((prev) => Math.min(prev + 1, maxScroll));
      setAutoScroll(false);
    }
    if (key.shift && key.downArrow) {
      setScrollOffset((prev) => {
        const next = Math.max(prev - 1, 0);
        if (next === 0) setAutoScroll(true);
        return next;
      });
    }
    if (key.pageUp) {
      setScrollOffset((prev) => Math.min(prev + 10, maxScroll));
      setAutoScroll(false);
    }
    if (key.pageDown) {
      setScrollOffset((prev) => {
        const next = Math.max(prev - 10, 0);
        if (next === 0) setAutoScroll(true);
        return next;
      });
    }
    // Meta+Up → scroll to top (oldest messages)
    if (key.meta && key.upArrow) {
      setScrollOffset(maxScroll);
      setAutoScroll(false);
    }
    // Meta+Down → scroll to bottom (newest messages, re-enable auto-scroll)
    if (key.meta && key.downArrow) {
      setScrollOffset(0);
      setAutoScroll(true);
    }
  });

  // Slice messages: endIndex tracks how many messages to show (skip the last
  // scrollOffset messages when the user has scrolled up).
  const endIndex = messages.length - scrollOffset;
  const visibleMessages = messages.slice(0, Math.max(endIndex, 0));

  return (
    <Box flexDirection="column" height={height} overflow="hidden">
      <Banner version={version} />
      {visibleMessages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      {scrollOffset > 0 && (
        <Box>
          <Text dimColor color="#888">
            {'↑ Shift+↑/↓ 스크롤 · PgUp/PgDn×10 · Meta+↑ 처음 · Meta+↓ 끝 · '}{scrollOffset}{'개 이전 메시지'}
          </Text>
        </Box>
      )}
    </Box>
  );
}
