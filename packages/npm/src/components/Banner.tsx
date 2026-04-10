import React, { useRef } from 'react';
import { Box, Text, useStdout } from 'ink';
import { THEME_COLORS, getBaseUrl } from '../config.js';

// Full 'G' block art for Tier 1 wide layout
const LOGO_ART = [
  '  ██████╗ ',
  ' ██╔════╝ ',
  ' ██║ ████╗',
  ' ██║ ╚═██║',
  ' ╚██████╔╝',
  '  ╚═════╝ ',
];

// Two-line block art for "GovOn" — used in Tier 2 and Tier 3-B
const TINY_GOVON = [
  '█▀▀ █▀█ █ █ █▀█ █▄ █',
  '█▄█ █▄█ ▀▄▀ █▄█ █ ▀█',
];

// Two-line block art for "G" only — used in Tier 3-A
const TINY_G = ['█▀▀', '█▄█'];

interface BannerProps {
  version: string;
}

/** Derive a short human-readable label for the current backend target. */
function getModeLabel(baseUrl: string): string {
  try {
    const url = new URL(baseUrl);
    const isLocal = url.hostname === '127.0.0.1' || url.hostname === 'localhost';
    return isLocal ? '로컬 모드' : `원격: ${url.hostname}`;
  } catch {
    return '로컬 모드';
  }
}

// ─── Tier 1: wide box layout (≥80 cols) ──────────────────────────────────────

function Tier1Banner({ version, modeLabel }: { version: string; modeLabel: string }) {
  return (
    <Box
      borderStyle="round"
      borderColor={THEME_COLORS.primary}
      paddingX={2}
      paddingY={1}
      flexDirection="row"
    >
      {/* Left column: big G logo + version + mode */}
      <Box flexDirection="column" marginRight={2}>
        {LOGO_ART.map((line, i) => (
          <Text key={i} color={THEME_COLORS.accent}>
            {line}
          </Text>
        ))}
        <Text> </Text>
        <Text bold color={THEME_COLORS.accent}>
          GovOn v{version}
        </Text>
        <Text color={THEME_COLORS.muted}>{modeLabel}</Text>
      </Box>

      {/* Vertical divider */}
      <Box
        borderStyle="single"
        borderLeft
        borderRight={false}
        borderTop={false}
        borderBottom={false}
        borderColor={THEME_COLORS.primary}
        marginRight={2}
      />

      {/* Right column: onboarding guide */}
      <Box flexDirection="column">
        <Text color={THEME_COLORS.muted}>
          질문을 입력하면 AI가 분석하고 도구를 실행합니다
        </Text>
        <Text color={THEME_COLORS.muted}>
          승인 필요 도구는 실행 전 확인합니다
        </Text>
        <Text> </Text>
        <Text>
          <Text color={THEME_COLORS.dimmed}>/help</Text>
          <Text color={THEME_COLORS.muted}>  명령어 목록    </Text>
          <Text color={THEME_COLORS.dimmed}>Esc</Text>
          <Text color={THEME_COLORS.muted}>     작업 취소</Text>
        </Text>
        <Text>
          <Text color={THEME_COLORS.dimmed}>/exit</Text>
          <Text color={THEME_COLORS.muted}>  종료          </Text>
          <Text color={THEME_COLORS.dimmed}>Ctrl+D</Text>
          <Text color={THEME_COLORS.muted}>  즉시 종료</Text>
        </Text>
        <Text> </Text>
        <Text color={THEME_COLORS.muted}>
          예시: &quot;민원 통계 알려줘&quot;
        </Text>
        <Text color={THEME_COLORS.muted}>
          {'      '}
          &quot;최근 승인 대기 건 조회&quot;
        </Text>
      </Box>
    </Box>
  );
}

// ─── Tier 2: medium layout (40–79 cols) ──────────────────────────────────────

function Tier2Banner({ version, modeLabel }: { version: string; modeLabel: string }) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box flexDirection="row">
        <Text color={THEME_COLORS.accent}>{TINY_GOVON[0]}</Text>
        <Text>{'   '}</Text>
        <Text bold color={THEME_COLORS.accent}>
          v{version}
        </Text>
        <Text color={THEME_COLORS.muted}> · {modeLabel}</Text>
      </Box>
      <Box flexDirection="row">
        <Text color={THEME_COLORS.accent}>{TINY_GOVON[1]}</Text>
        <Text>{'   '}</Text>
        <Text color={THEME_COLORS.dimmed}>/help</Text>
        <Text color={THEME_COLORS.muted}> · </Text>
        <Text color={THEME_COLORS.dimmed}>Esc</Text>
        <Text color={THEME_COLORS.muted}> · </Text>
        <Text color={THEME_COLORS.dimmed}>Ctrl+D</Text>
      </Box>
    </Box>
  );
}

// ─── Tier 3-A: narrow layout variant A (tiny G + info beside) ────────────────

function Tier3ABanner({ version, modeLabel }: { version: string; modeLabel: string }) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box flexDirection="row">
        <Text color={THEME_COLORS.accent}>{TINY_G[0]}</Text>
        <Text>{'  '}</Text>
        <Text bold color={THEME_COLORS.accent}>
          GovOn v{version}
        </Text>
      </Box>
      <Box flexDirection="row">
        <Text color={THEME_COLORS.accent}>{TINY_G[1]}</Text>
        <Text>{'  '}</Text>
        <Text color={THEME_COLORS.muted}>{modeLabel} · </Text>
        <Text color={THEME_COLORS.dimmed}>/help</Text>
        <Text color={THEME_COLORS.muted}> · </Text>
        <Text color={THEME_COLORS.dimmed}>Ctrl+D</Text>
      </Box>
    </Box>
  );
}

// ─── Tier 3-B: narrow layout variant B (tiny GovOn + info below) ─────────────

function Tier3BBanner({ version, modeLabel }: { version: string; modeLabel: string }) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color={THEME_COLORS.accent}>{TINY_GOVON[0]}</Text>
      <Text color={THEME_COLORS.accent}>{TINY_GOVON[1]}</Text>
      <Text>
        <Text bold color={THEME_COLORS.accent}>
          v{version}
        </Text>
        <Text color={THEME_COLORS.muted}> · {modeLabel}</Text>
      </Text>
      <Text>
        <Text color={THEME_COLORS.dimmed}>/help</Text>
        <Text color={THEME_COLORS.muted}> · </Text>
        <Text color={THEME_COLORS.dimmed}>Esc</Text>
        <Text color={THEME_COLORS.muted}> · </Text>
        <Text color={THEME_COLORS.dimmed}>Ctrl+D</Text>
      </Text>
    </Box>
  );
}

// ─── Main Banner ──────────────────────────────────────────────────────────────

export function Banner({ version }: BannerProps) {
  const { stdout } = useStdout();
  const cols = stdout?.columns ?? 80;

  const baseUrl = getBaseUrl();
  const modeLabel = getModeLabel(baseUrl);

  // Stable random variant for Tier 3 — fixed on mount, not re-randomized on re-render
  const useVariantA = useRef(Math.random() < 0.5).current;

  if (cols >= 80) {
    return <Tier1Banner version={version} modeLabel={modeLabel} />;
  }

  if (cols >= 40) {
    return <Tier2Banner version={version} modeLabel={modeLabel} />;
  }

  // Tier 3: <40 cols — randomly choose A or B
  return useVariantA ? (
    <Tier3ABanner version={version} modeLabel={modeLabel} />
  ) : (
    <Tier3BBanner version={version} modeLabel={modeLabel} />
  );
}
