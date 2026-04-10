/**
 * useTheme — resolves brand color tokens from the runtime environment.
 *
 * Respects the NO_COLOR convention (https://no-color.org/):
 * when NO_COLOR is set to any value (including empty string), all color
 * tokens are returned as empty strings so callers can safely pass them
 * to Ink's <Text color="..."> prop without emitting ANSI sequences.
 *
 * The returned object is stable across renders (memoised with no deps)
 * because process.env values never change at runtime.
 */

import { useMemo } from 'react';
import { THEME_COLORS } from '../config.js';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/** Brand color tokens used throughout the TUI. */
export interface ThemeTokens {
  /** Primary brand green — main interactive elements. */
  primary: string;
  /** Accent warm sand — highlights and callouts. */
  accent: string;
  /** Muted sage — secondary text and borders. */
  muted: string;
  /** Error red — destructive actions and error messages. */
  error: string;
  /** Warning amber — non-critical warnings. */
  warning: string;
  /** Success green — confirmation and done states. */
  success: string;
  /** Info blue — informational badges and labels. */
  info: string;
  /** Dimmed grey — disabled states and placeholders. */
  dimmed: string;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Returns resolved theme color tokens for the current runtime environment.
 *
 * Usage:
 *   const theme = useTheme();
 *   <Text color={theme.primary}>Hello</Text>
 */
export function useTheme(): ThemeTokens {
  return useMemo(() => {
    // NO_COLOR env var present → strip all color output regardless of value.
    if (process.env.NO_COLOR !== undefined) {
      return {
        primary: '',
        accent: '',
        muted: '',
        error: '',
        warning: '',
        success: '',
        info: '',
        dimmed: '',
      };
    }

    return { ...THEME_COLORS };
  }, []);
}
