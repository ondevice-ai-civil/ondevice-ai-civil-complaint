import { Writable } from 'node:stream';
import {
  ENTER_ALT_SCREEN,
  EXIT_ALT_SCREEN,
  BSU,
  ESU,
  ERASE_SCREEN,
  CURSOR_HOME,
  HIDE_CURSOR,
  SHOW_CURSOR,
  SYNC_OUTPUT_SUPPORTED,
} from './constants.js';

/**
 * RenderInterceptor wraps a real stdout stream and intercepts Ink's render output.
 *
 * Pattern (from Claude Code ink.tsx handleResize + onRender):
 *   resize → needsEraseBeforePaint = true   (deferred, no immediate write)
 *   render → if (needsEraseBeforePaint) prepend ERASE_SCREEN + CURSOR_HOME
 *           → wrap entire payload in BSU...ESU
 *           → single atomic realStdout.write()
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

  constructor(stdout: NodeJS.WriteStream) {
    this.realStdout = stdout;
    this.terminalRows = stdout.rows ?? 24;
    this.terminalCols = stdout.columns ?? 80;

    // Build a Writable that captures Ink's output without forwarding to real stdout
    const interceptor = this;
    const base = new Writable({
      write(chunk: Buffer | string, _encoding: BufferEncoding, callback: () => void) {
        interceptor.lastOutput = typeof chunk === 'string' ? chunk : chunk.toString();
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
   * Layout of the emitted buffer:
   *   [BSU]
   *   [ERASE_SCREEN + CURSOR_HOME]   ← only when needsEraseBeforePaint
   *   [CURSOR_HOME]                  ← always in alt-screen mode
   *   <Ink output>
   *   [ESU]
   */
  private flush(): void {
    if (!this.lastOutput) return;

    let buffer = '';

    if (SYNC_OUTPUT_SUPPORTED) buffer += BSU;

    if (this.needsEraseBeforePaint) {
      this.needsEraseBeforePaint = false;
      buffer += ERASE_SCREEN + CURSOR_HOME;
    }

    if (this.altScreenActive) {
      buffer += CURSOR_HOME;
    }

    buffer += this.lastOutput;

    if (SYNC_OUTPUT_SUPPORTED) buffer += ESU;

    // Single atomic write — eliminates partial-frame flicker
    this.realStdout.write(buffer);
    this.lastOutput = '';
  }
}
