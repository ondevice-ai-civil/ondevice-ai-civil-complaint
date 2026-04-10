import React, { useState, useCallback, useEffect } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';
import { THEME_COLORS } from '../config.js';

interface InputBarProps {
  onSubmit: (query: string) => void;
  /** When true, the input field is read-only and shows a busy placeholder. */
  disabled?: boolean;
}

export function InputBar({ onSubmit, disabled = false }: InputBarProps) {
  const [value, setValue] = useState('');

  // Clear stale input when entering disabled state
  useEffect(() => {
    if (disabled) setValue('');
  }, [disabled]);

  // Block input changes while disabled
  const handleChange = useCallback(
    (next: string) => {
      if (!disabled) setValue(next);
    },
    [disabled],
  );

  const handleSubmit = useCallback(
    (val: string) => {
      const trimmed = val.trim();
      if (!trimmed || disabled) return;
      onSubmit(trimmed);
      setValue('');
    },
    [onSubmit, disabled],
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
