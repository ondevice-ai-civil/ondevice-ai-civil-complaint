'use strict';

const { execSync, spawnSync } = require('child_process');

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 10;

/**
 * List of Python executable candidates in priority order
 */
const PYTHON_CANDIDATES = ['python3', 'python'];

/**
 * Returns the version of the given Python executable.
 * Returns null if the executable cannot be run or version parsing fails.
 * @param {string} cmd - Python command to execute
 * @returns {{ major: number, minor: number } | null}
 */
function getPythonVersion(cmd) {
  try {
    const result = spawnSync(cmd, ['--version'], { encoding: 'utf8', timeout: 5000 });
    if (result.status !== 0 || result.error) return null;

    // Output may appear as "Python 3.11.4" or on stderr (Python 2)
    const output = (result.stdout || result.stderr || '').trim();
    const match = output.match(/Python\s+(\d+)\.(\d+)/i);
    if (!match) return null;

    return { major: parseInt(match[1], 10), minor: parseInt(match[2], 10) };
  } catch {
    return null;
  }
}

/**
 * Finds a Python 3.10+ executable available on the system.
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
 * Checks whether the `govon` CLI is installed in PATH.
 * Looks for the Python govon binary, not the npm govon.js wrapper.
 * @returns {boolean}
 */
function isGovonInstalled() {
  try {
    // Verify govon path via which/where to confirm it is a Python binary
    const whichCmd = process.platform === 'win32' ? 'where' : 'which';
    const which = spawnSync(whichCmd, ['govon'], { encoding: 'utf8', timeout: 5000 });
    if (which.error || which.status !== 0) return false;

    const govonPath = (which.stdout || '').trim().split('\n')[0];
    // If the path is under node_modules/.bin it is the npm wrapper — ignore it
    if (govonPath.includes('node_modules')) return false;

    // Confirm directly via Python module
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
 * Inspects the entire Python environment and returns the result.
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
 * Prints the environment check result to stdout and displays guidance if issues are found.
 * @returns {boolean} - true if all conditions are met
 */
function printEnvironmentStatus() {
  const { pythonFound, pythonCmd, pythonVersion, govonInstalled } = checkEnvironment();

  if (!pythonFound) {
    console.error(
      [
        '',
        '  [govon] Python 3.10 or later is required.',
        '',
        '  Please install Python and try again:',
        '    https://www.python.org/downloads/',
        '',
        '  Or install via a package manager:',
        '    macOS:   brew install python@3.12',
        '    Ubuntu:  sudo apt install python3.12',
        '    Windows: winget install Python.Python.3.12',
        '',
      ].join('\n')
    );
    return false;
  }

  if (!govonInstalled) {
    console.error(
      [
        '',
        `  [govon] govon CLI is not installed. (Python ${pythonVersion} detected)`,
        '',
        '  Install it with:',
        `    ${pythonCmd} -m pip install govon`,
        '',
        '  If you are using a virtual environment:',
        '    python -m venv .venv && source .venv/bin/activate',
        `    pip install govon`,
        '',
      ].join('\n')
    );
    return false;
  }

  return true;
}

module.exports = {
  findPython,
  isGovonInstalled,
  checkEnvironment,
  printEnvironmentStatus,
};
