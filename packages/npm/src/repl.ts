/**
 * repl.ts — REPL-style TUI prototype for GovOn.
 *
 * Replaces Ink entirely with direct terminal control (claw-code style).
 * No React, no JSX. Pure Node.js readline + chalk + marked.
 *
 * Loop:
 *   1. Print banner
 *   2. Wait for server (poll health)
 *   3. Readline prompt
 *   4. Spinner while processing
 *   5. Stream response with inline markdown rendering
 *   6. Repeat from 3
 */

import * as readline from 'node:readline';
import { createRequire } from 'node:module';
import { Chalk } from 'chalk';
import { Marked } from 'marked';
import { markedTerminal } from 'marked-terminal';

import { createClient, isMockMode } from './clientFactory.js';
import { getBaseUrl, THEME_COLORS } from './config.js';
import type { IClient } from './client.interface.js';
import type { V3SSEEvent } from './types.js';
import {
  ERASE_SCREEN,
  CURSOR_HOME,
} from './terminal/constants.js';

const require = createRequire(import.meta.url);
const pkg = require('../package.json') as { version: string };

const chalk = new Chalk({ level: 3 });

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

function renderMarkdown(text: string, width: number): string {
  const md = new Marked(
    markedTerminal({
      reflowText: true,
      width,
      showSectionPrefix: false,
      strong: chalk.bold,
      em: chalk.italic,
      codespan: chalk.cyan,
      del: chalk.strikethrough,
    }),
  );
  const result = md.parse(text);
  return typeof result === 'string' ? result.trimEnd() : text;
}

// ---------------------------------------------------------------------------
// Strip ANSI from untrusted input
// ---------------------------------------------------------------------------

function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
}

// ---------------------------------------------------------------------------
// Spinner — single-line in-place update (claw-code style)
// ---------------------------------------------------------------------------

class TerminalSpinner {
  private readonly frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  private frameIndex = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private label: string;

  constructor(label: string) {
    this.label = label;
  }

  start(): void {
    this.timer = setInterval(() => {
      const frame = this.frames[this.frameIndex % this.frames.length];
      this.frameIndex++;
      // Save cursor → column 0 → clear line → print → restore cursor
      process.stdout.write(
        `\x1b7\x1b[G\x1b[2K  ${chalk.hex(THEME_COLORS.accent)(frame)} ${chalk.hex(THEME_COLORS.muted)(this.label)}\x1b8`,
      );
    }, 80);
  }

  update(label: string): void {
    this.label = label;
  }

  stop(finalLabel?: string): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    // Clear the spinner line
    process.stdout.write('\x1b[G\x1b[2K');
    if (finalLabel) {
      process.stdout.write(
        `  ${chalk.hex(THEME_COLORS.success)('✓')} ${chalk.hex(THEME_COLORS.muted)(finalLabel)}\n`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

function printBanner(version: string): void {
  const cols = process.stdout.columns ?? 80;
  const line = chalk.hex(THEME_COLORS.primary)('─'.repeat(Math.min(cols - 1, 60)));

  console.log();
  console.log(`  ${chalk.bold.hex(THEME_COLORS.accent)('GovOn')} ${chalk.hex(THEME_COLORS.muted)(`v${version}`)}`);
  console.log(`  ${line}`);
  console.log(`  ${chalk.hex(THEME_COLORS.muted)('질문을 입력하면 AI가 분석하고 도구를 실행합니다')}`);
  console.log(
    `  ${chalk.hex(THEME_COLORS.dimmed)('Ctrl+D')} ${chalk.hex(THEME_COLORS.muted)('종료')}  ` +
    `${chalk.hex(THEME_COLORS.dimmed)('Ctrl+C')} ${chalk.hex(THEME_COLORS.muted)('취소')}`,
  );
  if (isMockMode) {
    console.log(`  ${chalk.hex(THEME_COLORS.warning)('[MOCK MODE]')}`);
  }
  console.log();
}

// ---------------------------------------------------------------------------
// Wait for server to become healthy
// ---------------------------------------------------------------------------

async function waitForServer(client: IClient): Promise<void> {
  const spinner = new TerminalSpinner('서버 연결 중…');
  spinner.start();
  try {
    await client.health();
    spinner.stop('서버 연결 완료');
  } catch {
    spinner.update('서버 대기 중 (cold start)…');
    const ok = await client.waitForReady();
    if (!ok) {
      spinner.stop();
      console.error(chalk.hex(THEME_COLORS.error)('  서버 연결 실패'));
      process.exit(1);
    }
    spinner.stop('서버 연결 완료');
  }
}

// ---------------------------------------------------------------------------
// Stream response: v3 SSE → blocking fallback
// ---------------------------------------------------------------------------

interface StreamResult {
  text: string;
  sessionId: string | null;
}

async function streamResponse(
  client: IClient,
  query: string,
  sessionId: string | null,
): Promise<StreamResult> {
  const controller = new AbortController();
  const cols = process.stdout.columns ?? 80;

  // Forward Ctrl+C to abort the in-flight request
  const sigintHandler = (): void => {
    controller.abort();
  };
  process.on('SIGINT', sigintHandler);

  const spinner = new TerminalSpinner('생각 중…');
  spinner.start();

  let finalText = '';
  let newSessionId = sessionId;
  let streamStarted = false;

  try {
    for await (const event of client.streamV3(query, sessionId ?? undefined, undefined, controller.signal)) {
      if (controller.signal.aborted) break;

      const e = event as V3SSEEvent;

      switch (e.type) {
        case 'thinking_start':
          spinner.update(`분석 중 (iteration ${e.iteration})…`);
          break;

        case 'thinking_delta':
          // Thinking tokens — keep spinner only, skip rendering
          break;

        case 'tool_start':
          spinner.update(`도구 실행: ${e.tool}…`);
          break;

        case 'tool_end':
          spinner.update(e.success ? `${e.tool} ✓` : `${e.tool} ✗`);
          break;

        case 'response_delta':
          if (!streamStarted) {
            spinner.stop();
            streamStarted = true;
            process.stdout.write(
              `\n  ${chalk.bold.hex(THEME_COLORS.accent)('GovOn')}\n\n`,
            );
          }
          // Write raw streaming tokens (re-render after completion is impractical in prototype)
          process.stdout.write(e.content);
          finalText += e.content;
          break;

        case 'run_complete':
          if (!streamStarted) spinner.stop();
          finalText = e.text;
          newSessionId = e.session_id;
          break;

        case 'error':
          spinner.stop();
          console.error(`\n  ${chalk.hex(THEME_COLORS.error)('오류:')} ${e.error}`);
          break;
      }
    }
  } catch (err) {
    if (!controller.signal.aborted) {
      // v3 stream failed — fall back to blocking /v2/agent/run
      spinner.update('응답 대기 중…');
      try {
        const result = await client.run(query, sessionId ?? undefined, controller.signal);
        spinner.stop();
        finalText = result.text;
        newSessionId = result.session_id;
      } catch (blockErr) {
        spinner.stop();
        const msg = blockErr instanceof Error ? blockErr.message : String(blockErr);
        console.error(`\n  ${chalk.hex(THEME_COLORS.error)('오류:')} ${msg}`);
      }
    } else {
      spinner.stop();
    }
  } finally {
    process.off('SIGINT', sigintHandler);
  }

  if (finalText && streamStarted) {
    // Streaming path: raw tokens were already printed, add trailing newline
    process.stdout.write('\n');
  } else if (finalText && !streamStarted) {
    // Blocking fallback path: render full response as formatted markdown
    process.stdout.write(`\n  ${chalk.bold.hex(THEME_COLORS.accent)('GovOn')}\n\n`);
    process.stdout.write(renderMarkdown(stripAnsi(finalText), cols - 4));
    process.stdout.write('\n');
  }

  return { text: finalText, sessionId: newSessionId };
}

// ---------------------------------------------------------------------------
// REPL main loop
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Handle --version / -v flag
  if (args.includes('-v') || args.includes('--version')) {
    process.stdout.write(pkg.version + '\n');
    process.exit(0);
  }

  const baseUrl = getBaseUrl();
  const client = createClient(baseUrl);

  printBanner(pkg.version);
  await waitForServer(client);

  let sessionId: string | null = null;

  // Readline interface — streams through stdin/stdout with terminal prompt support
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `  ${chalk.bold.hex(THEME_COLORS.accent)('❯')} `,
    terminal: true,
  });

  // Process an optional positional argument as the initial query
  const initialQuery = args.find((a) => !a.startsWith('-'));
  if (initialQuery) {
    console.log(`  ${chalk.hex(THEME_COLORS.muted)('>')} ${initialQuery}`);
    const result = await streamResponse(client, initialQuery, sessionId);
    sessionId = result.sessionId;
    console.log();
  }

  rl.prompt();

  rl.on('line', async (line: string) => {
    const trimmed = line.trim();

    if (!trimmed) {
      rl.prompt();
      return;
    }

    // Built-in REPL commands
    if (trimmed === '/exit' || trimmed === '/quit') {
      rl.close();
      return;
    }

    if (trimmed === '/help') {
      console.log();
      console.log(`  ${chalk.hex(THEME_COLORS.dimmed)('/help')}   ${chalk.hex(THEME_COLORS.muted)('명령어 목록')}`);
      console.log(`  ${chalk.hex(THEME_COLORS.dimmed)('/exit')}   ${chalk.hex(THEME_COLORS.muted)('종료')}`);
      console.log(`  ${chalk.hex(THEME_COLORS.dimmed)('/clear')}  ${chalk.hex(THEME_COLORS.muted)('화면 지우기')}`);
      console.log(`  ${chalk.hex(THEME_COLORS.dimmed)('/new')}    ${chalk.hex(THEME_COLORS.muted)('새 세션 시작')}`);
      console.log();
      rl.prompt();
      return;
    }

    if (trimmed === '/clear') {
      process.stdout.write(ERASE_SCREEN + CURSOR_HOME);
      printBanner(pkg.version);
      rl.prompt();
      return;
    }

    if (trimmed === '/new') {
      sessionId = null;
      console.log(`  ${chalk.hex(THEME_COLORS.success)('✓')} ${chalk.hex(THEME_COLORS.muted)('새 세션을 시작합니다.')}`);
      console.log();
      rl.prompt();
      return;
    }

    const result = await streamResponse(client, trimmed, sessionId);
    sessionId = result.sessionId;
    console.log();
    rl.prompt();
  });

  rl.on('close', () => {
    console.log(`\n  ${chalk.hex(THEME_COLORS.muted)('종료합니다.')}`);
    process.exit(0);
  });

  // Ctrl+C at an empty prompt exits cleanly
  rl.on('SIGINT', () => {
    rl.close();
  });
}

main().catch((err: unknown) => {
  const msg = err instanceof Error ? err.message : String(err);
  console.error(chalk.hex(THEME_COLORS.error)(`Fatal: ${msg}`));
  process.exit(1);
});
