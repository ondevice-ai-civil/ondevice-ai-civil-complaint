/**
 * Terminal control layer — alternate screen, deferred erase, synchronized output.
 */

export { RenderInterceptor } from './RenderInterceptor.js';
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
