'use strict';

const { execSync, spawnSync } = require('child_process');

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 10;

/**
 * Python 실행 파일 후보 목록 (우선순위 순)
 */
const PYTHON_CANDIDATES = ['python3', 'python'];

/**
 * 주어진 python 실행 파일의 버전을 반환합니다.
 * 실행 불가능하거나 파싱 실패 시 null을 반환합니다.
 * @param {string} cmd - 실행할 python 명령어
 * @returns {{ major: number, minor: number } | null}
 */
function getPythonVersion(cmd) {
  try {
    const result = spawnSync(cmd, ['--version'], { encoding: 'utf8', timeout: 5000 });
    if (result.status !== 0 || result.error) return null;

    // "Python 3.11.4" 또는 stderr에 출력될 수 있음 (Python 2)
    const output = (result.stdout || result.stderr || '').trim();
    const match = output.match(/Python\s+(\d+)\.(\d+)/i);
    if (!match) return null;

    return { major: parseInt(match[1], 10), minor: parseInt(match[2], 10) };
  } catch {
    return null;
  }
}

/**
 * 시스템에서 사용 가능한 Python 3.10+ 실행 파일을 찾습니다.
 * @returns {{ cmd: string, major: number, minor: number } | null}
 */
function findPython() {
  for (const candidate of PYTHON_CANDIDATES) {
    const version = getPythonVersion(candidate);
    if (
      version &&
      (version.major > MIN_PYTHON_MAJOR ||
        (version.major === MIN_PYTHON_MAJOR && version.minor >= MIN_PYTHON_MINOR))
    ) {
      return { cmd: candidate, ...version };
    }
  }
  return null;
}

/**
 * `govon` CLI가 PATH에 설치되어 있는지 확인합니다.
 * npm의 govon.js wrapper가 아닌 Python의 govon binary를 찾습니다.
 * @returns {boolean}
 */
function isGovonInstalled() {
  try {
    // which/where로 govon 경로를 확인하여 Python binary인지 검증
    const whichCmd = process.platform === 'win32' ? 'where' : 'which';
    const which = spawnSync(whichCmd, ['govon'], { encoding: 'utf8', timeout: 5000 });
    if (which.error || which.status !== 0) return false;

    const govonPath = (which.stdout || '').trim().split('\n')[0];
    // npm bin 경로(node_modules/.bin)에 있으면 npm wrapper이므로 무시
    if (govonPath.includes('node_modules')) return false;

    // Python module로 직접 확인
    const python = findPython();
    if (!python) return false;

    const result = spawnSync(python.cmd, ['-m', 'govon', '--version'], {
      encoding: 'utf8',
      timeout: 5000,
    });
    return !result.error && result.status === 0;
  } catch {
    return false;
  }
}

/**
 * Python 환경 전체를 검사하고 결과를 반환합니다.
 * @returns {{
 *   pythonFound: boolean,
 *   pythonCmd: string | null,
 *   pythonVersion: string | null,
 *   govonInstalled: boolean
 * }}
 */
function checkEnvironment() {
  const python = findPython();
  const govonInstalled = python ? isGovonInstalled() : false;

  return {
    pythonFound: python !== null,
    pythonCmd: python ? python.cmd : null,
    pythonVersion: python ? `${python.major}.${python.minor}` : null,
    govonInstalled,
  };
}

/**
 * Python govon 패키지를 pip로 자동 설치합니다.
 * @param {string} pythonCmd - python 실행 파일 경로
 * @returns {boolean} - 설치 성공 여부
 */
function autoInstallGovon(pythonCmd) {
  console.log('\n  [govon] Python govon 패키지를 자동으로 설치합니다…');
  console.log(`  → ${pythonCmd} -m pip install govon\n`);

  const result = spawnSync(pythonCmd, ['-m', 'pip', 'install', 'govon'], {
    stdio: 'inherit',
    timeout: 120000,
  });

  if (result.error || result.status !== 0) {
    console.error(
      [
        '',
        '  [govon] 자동 설치에 실패했습니다.',
        '',
        '  아래 명령어로 직접 설치해 주세요:',
        `    ${pythonCmd} -m pip install govon`,
        '',
        '  권한 문제 시:',
        `    ${pythonCmd} -m pip install --user govon`,
        '',
      ].join('\n')
    );
    return false;
  }

  console.log('\n  [govon] ✓ Python govon 패키지 설치 완료.\n');
  return true;
}

/**
 * 환경 검사 결과를 stdout에 출력하고 문제가 있으면 안내 메시지를 표시합니다.
 * Python govon이 없으면 자동 설치를 시도합니다.
 * @returns {boolean} - 모든 조건이 충족되면 true
 */
function printEnvironmentStatus() {
  const { pythonFound, pythonCmd, pythonVersion, govonInstalled } = checkEnvironment();

  if (!pythonFound) {
    console.error(
      [
        '',
        '  [govon] Python 3.10 이상이 필요합니다.',
        '',
        '  Python을 설치한 뒤 다시 시도해 주세요:',
        '    https://www.python.org/downloads/',
        '',
        '  또는 패키지 관리자를 이용하세요:',
        '    macOS:   brew install python@3.12',
        '    Ubuntu:  sudo apt install python3.12',
        '    Windows: winget install Python.Python.3.12',
        '',
      ].join('\n')
    );
    return false;
  }

  if (!govonInstalled) {
    // 자동 설치 시도
    if (!autoInstallGovon(pythonCmd)) {
      return false;
    }
    // 설치 후 재검증
    if (!isGovonInstalled()) {
      console.error(
        [
          '',
          '  [govon] 설치 후에도 govon CLI를 찾을 수 없습니다.',
          '',
          '  PATH에 pip 설치 경로가 포함되어 있는지 확인하세요.',
          `  또는 직접 실행: ${pythonCmd} -m govon`,
          '',
        ].join('\n')
      );
      return false;
    }
  }

  return true;
}

module.exports = {
  findPython,
  isGovonInstalled,
  autoInstallGovon,
  checkEnvironment,
  printEnvironmentStatus,
};
