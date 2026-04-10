import React from 'react';
import { Box, Text } from 'ink';
import { THEME_COLORS, getBaseUrl } from '../config.js';

// ASCII art for the 'G' logo, one line per row
const LOGO_ART = [
  '  ██████╗ ',
  ' ██╔════╝ ',
  ' ██║ ████╗',
  ' ██║ ╚═██║',
  ' ╚██████╔╝',
  '  ╚═════╝ ',
];

interface BannerProps {
  version: string;
}

export function Banner({ version }: BannerProps) {
  const baseUrl = getBaseUrl();

  // Derive a short label describing the current backend target
  let modeLabel: string;
  try {
    const url = new URL(baseUrl);
    const isLocal =
      url.hostname === '127.0.0.1' || url.hostname === 'localhost';
    modeLabel = isLocal ? '로컬 모드' : `원격: ${url.hostname}`;
  } catch {
    modeLabel = '로컬 모드';
  }

  return (
    <Box
      borderStyle="round"
      borderColor={THEME_COLORS.primary}
      paddingX={2}
      paddingY={1}
      flexDirection="row"
    >
      {/* Left column: logo art + version info */}
      <Box flexDirection="column" marginRight={3}>
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
        marginRight={3}
      />

      {/* Right column: onboarding tips */}
      <Box flexDirection="column">
        <Text bold color={THEME_COLORS.accent}>
          시작 가이드
        </Text>
        <Text color={THEME_COLORS.muted}>
          질문을 입력하면 AI 에이전트가 분석하고 도구를 사용합니다
        </Text>
        <Text color={THEME_COLORS.muted}>/help 로 명령어 목록 확인</Text>
        <Text color={THEME_COLORS.muted}>/exit 또는 Ctrl+D 로 종료</Text>
        <Text> </Text>
        <Text color={THEME_COLORS.muted}>
          승인 필요 도구는 실행 전 확인을 요청합니다
        </Text>
        <Text color={THEME_COLORS.muted}>Esc 로 진행 중인 작업 취소</Text>
      </Box>
    </Box>
  );
}
