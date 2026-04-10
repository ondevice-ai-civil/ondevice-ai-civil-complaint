import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { Marked } from 'marked';
import { markedTerminal } from 'marked-terminal';
import { Chalk } from 'chalk';
import { useTerminalSize } from '../contexts/index.js';

// Force chalk level 3 (full 16m color) regardless of tty detection.
// Ink takes over stdout which makes tty.isatty(1) === false, causing the
// internal chalk instance inside marked-terminal to fall back to level 0
// and strip all ANSI formatting (bold, italic, code spans, etc.).
const ink_chalk = new Chalk({ level: 3 });

interface MarkdownViewProps {
  /** Raw markdown text to render. */
  content: string;
  /** Whether more content is still streaming. */
  streaming?: boolean;
}

// Strip ANSI escape sequences from untrusted server content
function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
}

export function MarkdownView({ content, streaming = false }: MarkdownViewProps) {
  const { columns: width } = useTerminalSize();

  const rendered = useMemo(() => {
    if (!content) return '';
    try {
      // Create a fresh Marked instance with the current terminal width so that
      // re-renders after a terminal resize use the correct column count.
      const md = new Marked(
        markedTerminal({
          reflowText: true,
          width,
          showSectionPrefix: false,
          strong: ink_chalk.bold,
          em: ink_chalk.italic,
          codespan: ink_chalk.cyan,
          del: ink_chalk.strikethrough,
        }),
      );
      // marked.parse() is synchronous when the terminal renderer extension is active.
      const result = md.parse(stripAnsi(content));
      // Remove trailing newlines produced by marked's block-level output.
      return typeof result === 'string' ? result.trimEnd() : '';
    } catch {
      // Fall back to sanitized text so the UI never goes blank on a parse error.
      return stripAnsi(content);
    }
  }, [content, width]);

  if (!rendered) return null;

  return (
    <Box flexDirection="column">
      <Text wrap="wrap">{rendered}</Text>
      {streaming && <Text dimColor>{'▍'}</Text>}
    </Box>
  );
}
