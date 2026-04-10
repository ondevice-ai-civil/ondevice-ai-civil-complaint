import React, { useState, useCallback, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { THEME_COLORS } from '../config.js';

interface InputBarProps {
  onSubmit: (query: string) => void;
  /** When true, the input field is read-only and shows a busy placeholder. */
  disabled?: boolean;
  /**
   * Navigate through history. Returns the entry string at that position,
   * '' when returning to current (empty) input, or undefined if nothing
   * to navigate.
   */
  onNavigate?: (direction: 'up' | 'down') => string | undefined;
  /** Reset the history cursor when the user starts typing fresh. */
  onResetCursor?: () => void;
}

export function InputBar({ onSubmit, disabled = false, onNavigate, onResetCursor }: InputBarProps) {
  const [value, setValue] = useState('');

  // Clear stale input when entering disabled state
  useEffect(() => {
    if (disabled) setValue('');
  }, [disabled]);

  // Block input changes while disabled
  const handleChange = useCallback(
    (next: string) => {
      if (!disabled) {
        // When the user types manually, reset the history cursor so that
        // subsequent Up/Down navigations start from the latest entry again.
        if (next !== value) onResetCursor?.();
        setValue(next);
      }
    },
    [disabled, value, onResetCursor],
  );

  const handleSubmit = useCallback(
    (val: string) => {
      const trimmed = val.trim();
      if (!trimmed || disabled) return;
      onResetCursor?.();
      onSubmit(trimmed);
      setValue('');
    },
    [onSubmit, disabled, onResetCursor],
  );

  // Up/Down arrow keys navigate through history.
  // Only intercept when input is empty or Up is pressed (matching shell behaviour:
  // Up always navigates; Down only navigates when cursor is inside history).
  useInput(
    (_input, key) => {
      if (disabled) return;
      if (key.upArrow) {
        const entry = onNavigate?.('up');
        if (entry !== undefined) setValue(entry);
      }
      if (key.downArrow) {
        const entry = onNavigate?.('down');
        if (entry !== undefined) setValue(entry);
      }
    },
    { isActive: !disabled },
  );

  return (
    <Box>
      <Text bold color={disabled ? 'gray' : THEME_COLORS.accent}>
        {'❯ '}
      </Text>
      <TextInput
        value={value}
        onChange={handleChange}
        onSubmit={handleSubmit}
        placeholder={disabled ? '처리 중…' : '질문을 입력하세요'}
      />
    </Box>
  );
}
