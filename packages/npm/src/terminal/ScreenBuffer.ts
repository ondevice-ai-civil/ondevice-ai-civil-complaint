/**
 * Double-buffered terminal screen with cell-level diffing.
 *
 * On each render cycle:
 *   1. write(ansiText)  — parse Ink output into curr buffer
 *   2. diff()           — generate minimal ANSI patch (changed cells only)
 *   3. swap()           — rotate curr → prev, reset curr for next frame
 */

import { Cell, EMPTY_CELL, parseAnsi } from './AnsiParser.js';
import { cursorPosition } from './constants.js';

/** SGR reset sequence */
const SGR_RESET = '\x1b[0m';

/** Compare two cells for visual equality. */
function cellsEqual(a: Cell, b: Cell): boolean {
  return a.char === b.char && a.style === b.style;
}

/**
 * Build the styled text string for a contiguous run of cells [start, end].
 * Emits style sequences only when the style changes, and appends SGR_RESET
 * at the end to leave the terminal in a clean state.
 */
function renderCellRange(row: Cell[], start: number, end: number): string {
  let out = '';
  let lastStyle = '';

  for (let c = start; c <= end; c++) {
    const cell = row[c];

    // Skip wide-char continuation cells — they are already written by the lead cell
    if (cell.width === 0) continue;

    if (cell.style !== lastStyle) {
      if (cell.style === '') {
        out += SGR_RESET;
      } else {
        out += cell.style;
      }
      lastStyle = cell.style;
    }

    out += cell.char === '' ? ' ' : cell.char;
  }

  // Always reset style after a patch to avoid colour bleed
  if (lastStyle !== '') out += SGR_RESET;

  return out;
}

export class ScreenBuffer {
  private prev: Cell[][];
  private curr: Cell[][];
  private cols: number;
  private rows: number;
  /** When true, next diff() call emits a full-frame repaint instead of a patch. */
  private needsFullRedraw = true;

  constructor(cols: number, rows: number) {
    this.cols = cols;
    this.rows = rows;
    this.prev = this.createEmptyGrid();
    this.curr = this.createEmptyGrid();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Parse ANSI-styled text and write the result into the current buffer.
   * Called once per render cycle before diff().
   */
  write(ansiText: string): void {
    this.curr = parseAnsi(ansiText, this.cols, this.rows);
  }

  /**
   * Generate the minimal ANSI sequence that transforms the previous frame
   * into the current frame, using cell-level diffing.
   *
   * Algorithm:
   *   For each row, skip if all cells are identical to the previous frame.
   *   Otherwise, find the leftmost and rightmost differing columns and emit
   *   a single cursorPosition + styled patch for that span.
   *
   * On the first call after construction or resize, emits a full repaint.
   *
   * @returns ANSI string ready to write to stdout.
   */
  diff(): string {
    if (this.needsFullRedraw) {
      this.needsFullRedraw = false;
      return this.renderFull();
    }

    let out = '';

    for (let r = 0; r < this.rows; r++) {
      const prevRow = this.prev[r];
      const currRow = this.curr[r];

      // Fast path: check if the entire row is unchanged
      let rowChanged = false;
      for (let c = 0; c < this.cols; c++) {
        if (!cellsEqual(prevRow[c], currRow[c])) {
          rowChanged = true;
          break;
        }
      }
      if (!rowChanged) continue;

      // Find the first differing column (left scan)
      let firstDiff = 0;
      while (firstDiff < this.cols && cellsEqual(prevRow[firstDiff], currRow[firstDiff])) {
        firstDiff++;
      }

      // Find the last differing column (right scan)
      let lastDiff = this.cols - 1;
      while (lastDiff > firstDiff && cellsEqual(prevRow[lastDiff], currRow[lastDiff])) {
        lastDiff--;
      }

      // Emit cursor move + styled patch for the changed span
      out += cursorPosition(r + 1, firstDiff + 1);
      out += renderCellRange(currRow, firstDiff, lastDiff);
    }

    return out;
  }

  /**
   * Rotate buffers: prev ← curr, then clear curr for the next frame.
   * Reuses existing arrays to avoid GC pressure.
   */
  swap(): void {
    const tmp = this.prev;
    this.prev = this.curr;
    this.curr = tmp;
    this.clearGrid(this.curr);
  }

  /**
   * Resize the buffers to new dimensions and schedule a full repaint.
   * Called when the terminal is resized.
   */
  resize(cols: number, rows: number): void {
    this.cols = cols;
    this.rows = rows;
    this.prev = this.createEmptyGrid();
    this.curr = this.createEmptyGrid();
    this.needsFullRedraw = true;
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /** Emit cursorPosition + full row content for every row in curr. */
  private renderFull(): string {
    let out = '';
    for (let r = 0; r < this.rows; r++) {
      out += cursorPosition(r + 1, 1);
      out += renderCellRange(this.curr[r], 0, this.cols - 1);
    }
    return out;
  }

  /** Allocate a fresh rows × cols grid of EMPTY_CELL copies. */
  private createEmptyGrid(): Cell[][] {
    return Array.from({ length: this.rows }, () =>
      Array.from({ length: this.cols }, () => ({ ...EMPTY_CELL })),
    );
  }

  /**
   * Reset every cell in an existing grid to EMPTY_CELL without reallocating.
   * Used in swap() to recycle the old prev array as the new curr buffer.
   */
  private clearGrid(grid: Cell[][]): void {
    for (let r = 0; r < grid.length; r++) {
      for (let c = 0; c < grid[r].length; c++) {
        grid[r][c] = { ...EMPTY_CELL };
      }
    }
  }
}
