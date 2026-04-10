import React, { useState, useCallback } from 'react';
import { Box, Text } from 'ink';
import TextInput from 'ink-text-input';

interface InputBarProps {
  onSubmit: (query: string) => void;
  /** When true, the input field is read-only and shows a busy placeholder. */
  disabled?: boolean;
}

export function InputBar({ onSubmit, disabled = false }: InputBarProps) {
  const [value, setValue] = useState('');

  const handleSubmit = useCallback(
    (val: string) => {
      const trimmed = val.trim();
      // Ignore empty submissions and block input while a request is in flight
      if (!trimmed || disabled) return;
      onSubmit(trimmed);
      setValue('');
    },
    [onSubmit, disabled],
  );

  return (
    <Box>
      {/* Prompt character mirrors the Python TUI green arrow */}
      <Text bold color={disabled ? 'gray' : 'green'}>
        {'❯ '}
      </Text>
      <TextInput
        value={value}
        onChange={setValue}
        onSubmit={handleSubmit}
        placeholder={disabled ? '처리 중…' : '질문을 입력하세요'}
      />
    </Box>
  );
}
