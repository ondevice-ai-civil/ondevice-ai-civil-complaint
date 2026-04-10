/**
 * Minimal type declarations for marked-terminal v7.
 * The package ships no bundled .d.ts and there is no @types/marked-terminal.
 */
declare module 'marked-terminal' {
  import type { MarkedExtension } from 'marked';

  interface MarkedTerminalOptions {
    reflowText?: boolean;
    width?: number;
    showSectionPrefix?: boolean;
    unescape?: boolean;
    emoji?: boolean;
    firstHeading?: unknown;
    tab?: number;
    tableOptions?: Record<string, unknown>;
  }

  /** Factory that returns a marked extension for terminal rendering. */
  export function markedTerminal(options?: MarkedTerminalOptions): MarkedExtension;

  /** Default export mirrors the named export for CJS interop. */
  export default markedTerminal;
}
