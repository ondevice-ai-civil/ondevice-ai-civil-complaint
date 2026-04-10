/**
 * Lightweight ANSI tokenizer that converts Ink's ANSI-styled output string
 * into a 2D cell grid for double-buffered diffing.
 */

export interface Cell {
  /** Single visible character, or ' ' for wide char continuation cells. */
  char: string;
  /** 0 = continuation (wide char right half), 1 = normal, 2 = wide (CJK/emoji) */
  width: number;
  /** Accumulated SGR sequence, e.g. '\x1b[1;31m'. Empty string = default style. */
  style: string;
}

/** Default empty cell used to fill unwritten grid positions. */
export const EMPTY_CELL: Cell = { char: ' ', width: 1, style: '' };

/**
 * Determine the display width of a Unicode code point.
 * Returns 2 for wide characters (CJK, emoji, fullwidth), 1 otherwise.
 *
 * Covered ranges (non-exhaustive but sufficient for most terminal content):
 *   U+1100-U+115F  Hangul Jamo
 *   U+2E80-U+303E  CJK Radicals, Kangxi, CJK Symbols
 *   U+3040-U+33FF  Japanese (Hiragana, Katakana, CJK Compatibility)
 *   U+3400-U+4DBF  CJK Extension A
 *   U+4E00-U+9FFF  CJK Unified Ideographs
 *   U+A000-U+A4CF  Yi Syllables / Radicals
 *   U+AC00-U+D7AF  Hangul Syllables
 *   U+F900-U+FAFF  CJK Compatibility Ideographs
 *   U+FE10-U+FE1F  Vertical Forms
 *   U+FE30-U+FE4F  CJK Compatibility Forms
 *   U+FF00-U+FF60  Fullwidth Latin / Halfwidth Katakana
 *   U+FFE0-U+FFE6  Fullwidth Signs
 *   U+1F300-U+1FAFF Emoji and Miscellaneous Symbols
 *   U+20000-U+2FFFD CJK Extension B-F
 *   U+30000-U+3FFFD CJK Extension G+
 */
function codePointWidth(cp: number): 1 | 2 {
  if (
    (cp >= 0x1100 && cp <= 0x115f) ||   // Hangul Jamo
    (cp >= 0x2e80 && cp <= 0x303e) ||   // CJK Radicals / Kangxi / Symbols
    (cp >= 0x3040 && cp <= 0x33ff) ||   // Japanese / CJK Compatibility
    (cp >= 0x3400 && cp <= 0x4dbf) ||   // CJK Extension A
    (cp >= 0x4e00 && cp <= 0x9fff) ||   // CJK Unified Ideographs
    (cp >= 0xa000 && cp <= 0xa4cf) ||   // Yi
    (cp >= 0xac00 && cp <= 0xd7af) ||   // Hangul Syllables
    (cp >= 0xf900 && cp <= 0xfaff) ||   // CJK Compatibility Ideographs
    (cp >= 0xfe10 && cp <= 0xfe1f) ||   // Vertical Forms
    (cp >= 0xfe30 && cp <= 0xfe4f) ||   // CJK Compatibility Forms
    (cp >= 0xff00 && cp <= 0xff60) ||   // Fullwidth / Halfwidth
    (cp >= 0xffe0 && cp <= 0xffe6) ||   // Fullwidth signs
    (cp >= 0x1f300 && cp <= 0x1faff) || // Emoji / Misc Symbols
    (cp >= 0x20000 && cp <= 0x2fffd) || // CJK Extension B-F
    (cp >= 0x30000 && cp <= 0x3fffd)    // CJK Extension G+
  ) {
    return 2;
  }
  return 1;
}

/**
 * Create a fresh rows × cols grid filled with EMPTY_CELL clones.
 * Each row is an independent array so callers can mutate rows in-place.
 */
function createGrid(cols: number, rows: number): Cell[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({ ...EMPTY_CELL })),
  );
}

/**
 * Parse ANSI-styled text into a rows × cols cell grid.
 *
 * State machine behaviour:
 *   - ESC '[' ... 'm'  → SGR sequence: accumulate into currentStyle
 *   - ESC '[' ... <X>  → non-SGR CSI: skip entirely
 *   - ESC <other>      → skip 2 characters (ESC + one byte)
 *   - '\n'             → advance row, reset column to 0
 *   - '\r'             → reset column to 0
 *   - printable char   → write to cell with currentStyle, advance by char width
 *   - '\x00'-'\x1f'   → control chars other than CR/LF: skip
 *
 * Cells beyond the grid boundaries are silently discarded.
 * Remaining unfilled cells keep their EMPTY_CELL initialisation.
 */
export function parseAnsi(text: string, cols: number, rows: number): Cell[][] {
  const grid = createGrid(cols, rows);

  let row = 0;
  let col = 0;
  let currentStyle = '';
  let i = 0;
  const len = text.length;

  while (i < len) {
    const ch = text[i];

    if (ch === '\x1b') {
      // Escape sequence
      const next = text[i + 1];

      if (next === '[') {
        // CSI sequence: collect up to the final byte (ASCII 0x40–0x7E)
        let j = i + 2;
        while (j < len && (text.charCodeAt(j) < 0x40 || text.charCodeAt(j) > 0x7e)) {
          j++;
        }
        const finalByte = text[j] ?? '';
        const body = text.slice(i + 2, j); // parameter bytes (without ESC[)

        if (finalByte === 'm') {
          // SGR — accumulate style string (preserve full escape for later emission)
          currentStyle = '\x1b[' + body + 'm';
          // Special case: SGR 0 (reset) — clear accumulated style
          if (body === '0' || body === '') {
            currentStyle = '';
          }
        }
        // Non-SGR CSI: skip entirely

        i = j + 1;
        continue;
      } else if (next === ']') {
        // OSC sequence: read until ST (ESC \) or BEL
        let j = i + 2;
        while (j < len) {
          if (text[j] === '\x07') { j++; break; }
          if (text[j] === '\x1b' && text[j + 1] === '\\') { j += 2; break; }
          j++;
        }
        i = j;
        continue;
      } else {
        // Other escape sequences: skip ESC + one byte
        i += 2;
        continue;
      }
    }

    if (ch === '\n') {
      row++;
      col = 0;
      i++;
      continue;
    }

    if (ch === '\r') {
      col = 0;
      i++;
      continue;
    }

    // Skip control characters (except CR/LF handled above)
    const code = ch.charCodeAt(0);
    if (code < 0x20) {
      i++;
      continue;
    }

    // Printable character (may be part of a surrogate pair)
    let cpChar = ch;
    let cp = ch.codePointAt(0) ?? code;

    // Handle surrogate pairs (emoji etc.)
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < len) {
      const low = text.charCodeAt(i + 1);
      if (low >= 0xdc00 && low <= 0xdfff) {
        cpChar = ch + text[i + 1];
        cp = text.codePointAt(i) ?? cp;
        i += 2;
      } else {
        i++;
      }
    } else {
      i++;
    }

    const w = codePointWidth(cp);

    if (row < rows && col < cols) {
      grid[row][col] = { char: cpChar, width: w, style: currentStyle };

      if (w === 2) {
        // Fill continuation cell to the right
        if (col + 1 < cols) {
          grid[row][col + 1] = { char: '', width: 0, style: currentStyle };
        }
        col += 2;
      } else {
        col += 1;
      }
    } else if (row >= rows) {
      // Overflow beyond grid height: stop parsing visible content
      // (style updates may still arrive, keep scanning for SGR)
    } else {
      // col >= cols: skip printable characters on this row until CR/LF
      col += w;
    }
  }

  return grid;
}
