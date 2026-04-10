import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { marked } from 'marked';
import { markedTerminal } from 'marked-terminal';

// Configure marked with the terminal renderer once at module load time.
// markedTerminal() returns a marked extension object compatible with the
// marked.use() API introduced in marked v5+ and still current in v15.
marked.use(
  markedTerminal({
    reflowText: true,
    width: process.stdout.columns || 80,
    showSectionPrefix: false,
  }),
);

interface MarkdownViewProps {
  /** Raw markdown text to render. */
  content: string;
  /** Whether more content is still streaming. */
  streaming?: boolean;
}

export function MarkdownView({ content, streaming = false }: MarkdownViewProps) {
  const rendered = useMemo(() => {
    if (!content) return '';
    try {
      // marked.parse() is synchronous when the terminal renderer extension is active.
      const result = marked.parse(content);
      // Remove trailing newlines produced by marked's block-level output.
      return typeof result === 'string' ? result.trimEnd() : '';
    } catch {
      // Fall back to raw text so the UI never goes blank on a parse error.
      return content;
    }
  }, [content]);

  if (!rendered) return null;

  return (
    <Box flexDirection="column">
      <Text wrap="wrap">{rendered}</Text>
      {streaming && <Text dimColor>{'▍'}</Text>}
    </Box>
  );
}
