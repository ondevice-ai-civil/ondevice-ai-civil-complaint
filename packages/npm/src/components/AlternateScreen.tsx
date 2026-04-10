/**
 * AlternateScreen.tsx — full-screen layout container.
 *
 * Provides a <Box> sized to the current terminal dimensions via
 * TerminalSizeContext. The actual alternate screen escape sequences
 * (ENTER_ALT_SCREEN / EXIT_ALT_SCREEN) are managed by RenderInterceptor
 * in index.tsx — this component is purely a layout primitive.
 */

import React from 'react';
import { Box } from 'ink';
import { useTerminalSize } from '../contexts/index.js';

interface AlternateScreenProps {
  children: React.ReactNode;
}

/**
 * Renders children inside a Box that fills the entire terminal viewport.
 * Dimensions are kept in sync with the terminal via TerminalSizeContext.
 */
export function AlternateScreen({ children }: AlternateScreenProps) {
  const { columns, rows } = useTerminalSize();
  return (
    <Box flexDirection="column" height={rows} width={columns} overflow="hidden">
      {children}
    </Box>
  );
}
