#!/usr/bin/env node
'use strict';

const { spawn } = require('child_process');
const { printEnvironmentStatus, isGovonInstalled } = require('../lib/python-check');

const args = process.argv.slice(2);

// postinstall 또는 직접 호출 시 환경 점검만 수행하고 종료
if (args[0] === '--check-install') {
  const ok = printEnvironmentStatus();
  process.exit(ok ? 0 : 1);
}

// 실제 CLI 실행 경로: 환경이 올바르지 않으면 안내 메시지를 출력하고 종료
if (!printEnvironmentStatus()) {
  process.exit(1);
}

// govon CLI를 실제로 실행
const child = spawn('govon', args, {
  stdio: 'inherit',
  shell: false,
});

child.on('error', (err) => {
  if (err.code === 'ENOENT') {
    console.error(
      [
        '',
        '  [govon] govon 명령어를 실행할 수 없습니다.',
        '  pip install govon 으로 설치되었는지 확인하세요.',
        '',
      ].join('\n')
    );
  } else {
    console.error(`[govon] 실행 오류: ${err.message}`);
  }
  process.exit(1);
});

child.on('close', (code) => {
  process.exit(code ?? 0);
});
