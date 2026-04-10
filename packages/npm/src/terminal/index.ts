/**
 * Terminal control layer — alternate screen, deferred erase, synchronized output,
 * double-buffered screen with cell-level diffing.
 */

export { RenderInterceptor } from './RenderInterceptor.js';
export { ScreenBuffer } from './ScreenBuffer.js';
export { parseAnsi, EMPTY_CELL } from './AnsiParser.js';
export type { Cell } from './AnsiParser.js';
export {
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
  cursorPosition,
  cursorMove,
} from './constants.js';
