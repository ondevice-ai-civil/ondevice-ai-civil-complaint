import { Writable } from 'node:stream';
import {
  ENTER_ALT_SCREEN,
  EXIT_ALT_SCREEN,
  BSU,
  ESU,
  ERASE_SCREEN,
  ERASE_TO_END_OF_SCREEN,
  CURSOR_HOME,
  HIDE_CURSOR,
  SHOW_CURSOR,
  SYNC_OUTPUT_SUPPORTED,
} from './constants.js';
import { ScreenBuffer } from './ScreenBuffer.js';

/**
 * Regex to strip Ink's log-update eraseLines sequences.
 *
 * log-update.js generates: (\x1b[1A\x1b[2K)* \x1b[G
 *   - \x1b[1A  = cursor up 1 line
 *   - \x1b[2K  = erase entire line
 *   - \x1b[G   = cursor to column 1
 *
 * These use a stale previousLineCount which causes ghost lines on resize.
 * We strip them and use CURSOR_HOME + ERASE_TO_END_OF_SCREEN instead.
 */
// eslint-disable-next-line no-control-regex
const ERASE_LINES_RE = /^(\x1b\[1A\x1b\[2K)*\x1b\[G/;

/**
 * Regex to strip Ink's clearTerminal sequence (used when outputHeight >= rows).
 * clearTerminal = \x1b[2J\x1b[3J\x1b[H  (erase screen + erase scrollback + cursor home)
 */
// eslint-disable-next-line no-control-regex
const CLEAR_TERMINAL_RE = /\x1b\[2J\x1b\[3J\x1b\[H/g;

/**
 * RenderInterceptor — capture Ink's render output, strip buggy erase sequences,
 * and repaint using alternate screen + CURSOR_HOME + ERASE_TO_END_OF_SCREEN.
 *
 * Strategy (adapted from Claude Code ink.tsx):
 *   1. Ink writes to our proxy stream (not real stdout)
 *   2. We strip log-update's eraseLines (stale line count — root cause of ghost lines)
 *   3. We prepend CURSOR_HOME (alt-screen: always paint from top-left)
 *   4. We append ERASE_TO_END_OF_SCREEN (clean up stale content below new frame)
 *   5. On resize: set needsEraseBeforePaint flag (deferred — NOT immediate erase)
 *   6. Next flush consumes flag → prepend ERASE_SCREEN + CURSOR_HOME instead
 *   7. Entire frame wrapped in BSU/ESU for atomic, flicker-free output
 */
export class RenderInterceptor {
  private realStdout: NodeJS.WriteStream;
  private proxyStream: Writable & { columns: number; rows: number; isTTY: boolean };

  /** Deferred erase flag — set on resize, consumed in flush() */
  private needsEraseBeforePaint = false;
  private altScreenActive = false;
  private terminalRows: number;
  private terminalCols: number;

  /** Last raw output captured from Ink; cleared after each flush */
  private lastOutput = '';

  /** Double-buffered screen for cell-level diff rendering */
  private screenBuffer: ScreenBuffer;

  constructor(stdout: NodeJS.WriteStream) {
    this.realStdout = stdout;
    this.terminalRows = stdout.rows ?? 24;
    this.terminalCols = stdout.columns ?? 80;
    this.screenBuffer = new ScreenBuffer(this.terminalCols, this.terminalRows);

    // Build a Writable that captures Ink's output, strips erase sequences, and flushes
    const interceptor = this;
    const base = new Writable({
      write(chunk: Buffer | string, _encoding: BufferEncoding, callback: () => void) {
        let output = typeof chunk === 'string' ? chunk : chunk.toString();

        // Strip Ink's buggy eraseLines sequences (stale previousLineCount)
        output = output.replace(ERASE_LINES_RE, '');

        // Strip Ink's clearTerminal fallback
        output = output.replace(CLEAR_TERMINAL_RE, '');

        interceptor.lastOutput = output;
        interceptor.flush();
        callback();
      },
    });

    // Proxy stdout-specific properties dynamically so they always reflect
    // the current terminal dimensions, not values frozen at construction time.
    const proxied = base as typeof base & { columns: number; rows: number; isTTY: boolean };

    Object.defineProperty(proxied, 'columns', {
      get: () => interceptor.realStdout.columns ?? 80,
      enumerable: true,
      configurable: true,
    });

    Object.defineProperty(proxied, 'rows', {
      get: () => interceptor.realStdout.rows ?? 24,
      enumerable: true,
      configurable: true,
    });

    Object.defineProperty(proxied, 'isTTY', {
      get: () => interceptor.realStdout.isTTY ?? true,
      enumerable: true,
      configurable: true,
    });

    Object.defineProperty(proxied, 'hasColors', {
      get: () =>
        typeof (interceptor.realStdout as NodeJS.WriteStream & { hasColors?: () => boolean }).hasColors === 'function'
          ? (interceptor.realStdout as NodeJS.WriteStream & { hasColors: () => boolean }).hasColors()
          : true,
      enumerable: true,
      configurable: true,
    });

    Object.defineProperty(proxied, 'getColorDepth', {
      get: () => {
        const rs = interceptor.realStdout as NodeJS.WriteStream & { getColorDepth?: () => number };
        return typeof rs.getColorDepth === 'function'
          ? rs.getColorDepth.bind(rs)
          : () => 8;
      },
      enumerable: true,
      configurable: true,
    });

    Object.defineProperty(proxied, 'getWindowSize', {
      get: () => {
        const rs = interceptor.realStdout as NodeJS.WriteStream & { getWindowSize?: () => [number, number] };
        return typeof rs.getWindowSize === 'function'
          ? rs.getWindowSize.bind(rs)
          : () => [interceptor.terminalCols, interceptor.terminalRows];
      },
      enumerable: true,
      configurable: true,
    });

    this.proxyStream = proxied as Writable & { columns: number; rows: number; isTTY: boolean };
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** The fake stdout to hand to Ink's render() call. */
  get stdout(): Writable & { columns: number; rows: number; isTTY: boolean } {
    return this.proxyStream;
  }

  /** Switch to alternate screen buffer and hide the cursor. */
  enterAltScreen(): void {
    this.altScreenActive = true;
    this.realStdout.write(ENTER_ALT_SCREEN + ERASE_SCREEN + CURSOR_HOME + HIDE_CURSOR);
  }

  /** Restore normal screen buffer and show the cursor. */
  exitAltScreen(): void {
    this.altScreenActive = false;
    this.realStdout.write(SHOW_CURSOR + EXIT_ALT_SCREEN);
  }

  /**
   * Handle terminal resize.
   *
   * DOES NOT write ERASE_SCREEN immediately — sets the deferred flag so that
   * the next flush() prepends the erase as part of the same atomic write.
   * Emits 'resize' on the proxy stream so Ink recalculates layout.
   */
  handleResize(): void {
    const cols = this.realStdout.columns ?? 80;
    const rows = this.realStdout.rows ?? 24;

    if (cols === this.terminalCols && rows === this.terminalRows) return;

    this.terminalCols = cols;
    this.terminalRows = rows;

    if (this.altScreenActive) {
      // Deferred erase — consumed atomically inside flush()
      this.needsEraseBeforePaint = true;
      // Resize the screen buffer so diff() will emit a full repaint next cycle
      this.screenBuffer.resize(cols, rows);
    }

    // Trigger Ink re-render with updated dimensions
    this.proxyStream.emit('resize');
  }

  /** Release resources and restore the terminal to a clean state. */
  destroy(): void {
    if (this.altScreenActive) this.exitAltScreen();
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Write captured Ink output to real stdout in a single atomic call.
   *
   * When in alt-screen mode, uses ScreenBuffer to diff previous vs current
   * frame and emits only the changed cells (cursor-move + styled text patches).
   *
   * Layout of the emitted buffer (alt-screen path):
   *   [BSU]                              ← begin synchronized update
   *   [ERASE_SCREEN + CURSOR_HOME]       ← only when needsEraseBeforePaint (resize)
   *   <diff patch from ScreenBuffer>     ← only changed cells
   *   [ESU]                              ← end synchronized update
   *
   * Fallback (non-alt-screen):
   *   [CURSOR_HOME] <full output> [ERASE_TO_END_OF_SCREEN]
   */
  private flush(): void {
    if (!this.lastOutput) return;

    let buffer = '';

    if (SYNC_OUTPUT_SUPPORTED) buffer += BSU;

    if (this.altScreenActive) {
      if (this.needsEraseBeforePaint) {
        this.needsEraseBeforePaint = false;
        // screenBuffer.resize() was already called in handleResize(),
        // which sets needsFullRedraw — so diff() will emit a full repaint.
        buffer += ERASE_SCREEN + CURSOR_HOME;
      }

      // Parse Ink output into cell grid, diff, and swap buffers
      this.screenBuffer.write(this.lastOutput);
      buffer += this.screenBuffer.diff();
      this.screenBuffer.swap();
    } else {
      // Non-alt-screen: fall back to the original full-repaint strategy
      if (this.needsEraseBeforePaint) {
        this.needsEraseBeforePaint = false;
        buffer += ERASE_SCREEN + CURSOR_HOME;
      } else {
        buffer += CURSOR_HOME;
      }

      buffer += this.lastOutput;
      buffer += ERASE_TO_END_OF_SCREEN;
    }

    if (SYNC_OUTPUT_SUPPORTED) buffer += ESU;

    // Single atomic write — eliminates partial-frame flicker
    this.realStdout.write(buffer);
    this.lastOutput = '';
  }
}
