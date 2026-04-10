/**
 * ANSI escape sequence constants for terminal control.
 * Adapted from Claude Code's termio/dec.ts and termio/csi.ts.
 */

// CSI prefix (Control Sequence Introducer)
export const CSI = '\x1b[';

// Separator for CSI parameters
export const SEP = ';';

// --- Helper functions ---

/**
 * Generate a CSI escape sequence.
 * Single arg: raw body appended directly to CSI.
 * Multiple args: params joined by ';' + final byte (last arg).
 */
export function csi(...args: string[]): string {
  if (args.length === 1) {
    return CSI + args[0];
  }
  const params = args.slice(0, -1);
  const finalByte = args[args.length - 1];
  return CSI + params.join(SEP) + finalByte;
}

/**
 * Move cursor to absolute position (1-indexed).
 * Generates CSI row ; col H.
 */
export function cursorPosition(row: number, col: number): string {
  return csi(String(row), String(col), 'H');
}

/**
 * Move cursor relative to current position.
 * Positive x = right, negative x = left.
 * Positive y = down, negative y = up.
 */
export function cursorMove(x: number, y: number): string {
  let seq = '';
  if (x > 0) seq += csi(String(x), 'C');
  else if (x < 0) seq += csi(String(-x), 'D');
  if (y > 0) seq += csi(String(y), 'B');
  else if (y < 0) seq += csi(String(-y), 'A');
  return seq;
}

// --- DEC Private Modes ---

/** Enter alternate screen buffer (CSI ? 1049 h) */
export const ENTER_ALT_SCREEN = CSI + '?1049h';

/** Exit alternate screen buffer (CSI ? 1049 l) */
export const EXIT_ALT_SCREEN = CSI + '?1049l';

/** Begin Synchronized Update — DEC 2026 (CSI ? 2026 h) */
export const BSU = CSI + '?2026h';

/** End Synchronized Update — DEC 2026 (CSI ? 2026 l) */
export const ESU = CSI + '?2026l';

// --- Screen Control ---

/** Erase entire screen (CSI 2 J) */
export const ERASE_SCREEN = csi('2J');

/** Erase scrollback buffer (CSI 3 J) */
export const ERASE_SCROLLBACK = csi('3J');

/** Move cursor to home position (CSI H) */
export const CURSOR_HOME = csi('H');

/** Hide the cursor (CSI ? 25 l) */
export const HIDE_CURSOR = CSI + '?25l';

/** Show the cursor (CSI ? 25 h) */
export const SHOW_CURSOR = CSI + '?25h';

/** Erase entire current line (CSI 2 K) */
export const ERASE_LINE = csi('2K');

// --- Terminal capability detection ---

/**
 * Detect whether the terminal supports DEC 2026 synchronized output.
 * Checks known terminals that implement the feature.
 * TMUX is excluded as it does not pass the mode through reliably.
 */
export function isSyncOutputSupported(): boolean {
  if (process.env.TMUX) return false;

  const termProgram = process.env.TERM_PROGRAM;
  const term = process.env.TERM;

  if (
    termProgram === 'iTerm.app' ||
    termProgram === 'WezTerm' ||
    termProgram === 'WarpTerminal' ||
    termProgram === 'ghostty' ||
    termProgram === 'contour' ||
    termProgram === 'vscode' ||
    termProgram === 'alacritty'
  ) return true;

  if (term?.includes('kitty') || process.env.KITTY_WINDOW_ID) return true;
  if (term === 'xterm-ghostty') return true;
  if (term?.startsWith('foot')) return true;
  if (term?.includes('alacritty')) return true;

  if (process.env.ZED_TERM) return true;
  if (process.env.WT_SESSION) return true;

  const vteVersion = process.env.VTE_VERSION;
  if (vteVersion) {
    const version = parseInt(vteVersion, 10);
    if (version >= 6800) return true;
  }

  return false;
}

/** Pre-computed flag: true if the current terminal supports synchronized output. */
export const SYNC_OUTPUT_SUPPORTED: boolean = isSyncOutputSupported();
