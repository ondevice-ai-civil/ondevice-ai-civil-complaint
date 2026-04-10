import React, { useMemo } from 'react';
import { Box, Text, useStdout } from 'ink';
import { Marked } from 'marked';
import { markedTerminal } from 'marked-terminal';

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
  const { stdout } = useStdout();
  const width = stdout?.columns ?? 80;

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
        }),
      );
      // marked.parse() is synchronous when the terminal renderer extension is active.
      const result = md.parse(stripAnsi(content));
      // Remove trailing newlines produced by marked's block-level output.
      return typeof result === 'string' ? result.trimEnd() : '';
    } catch {
      // Fall back to raw text so the UI never goes blank on a parse error.
      return content;
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
