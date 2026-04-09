'use strict';

const { execSync, spawnSync } = require('child_process');

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 10;

/**
 * List of Python executable candidates (in priority order)
 */
const PYTHON_CANDIDATES = ['python3', 'python'];

/**
 * Returns the version of the given python executable.
 * Returns null if the executable cannot be run or version parsing fails.
 * @param {string} cmd - python command to execute
 * @returns {{ major: number, minor: number } | null}
 */
function getPythonVersion(cmd) {
  try {
    const result = spawnSync(cmd, ['--version'], { encoding: 'utf8', timeout: 5000 });
    if (result.status !== 0 || result.error) return null;

    // "Python 3.11.4" or may be printed to stderr (Python 2)
    const output = (result.stdout || result.stderr || '').trim();
    const match = output.match(/Python\s+(\d+)\.(\d+)/i);
    if (!match) return null;

    return { major: parseInt(match[1], 10), minor: parseInt(match[2], 10) };
  } catch {
    return null;
  }
}

/**
 * Finds an available Python 3.10+ executable on the system.
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
    // Verify govon path via which/where to confirm it is the Python binary
    const whichCmd = process.platform === 'win32' ? 'where' : 'which';
    const which = spawnSync(whichCmd, ['govon'], { encoding: 'utf8', timeout: 5000 });
    if (which.error || which.status !== 0) return false;

    const govonPath = (which.stdout || '').trim().split('\n')[0];
    // If found in npm bin path (node_modules/.bin), it is the npm wrapper — skip it
    if (govonPath.includes('node_modules')) return false;

    // Verify directly as a Python module
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
 * Automatically installs the Python govon package via pip.
 * @param {string} pythonCmd - path to the python executable
 * @returns {boolean} - true if installation succeeded
 */
function autoInstallGovon(pythonCmd) {
  console.log('\n  [govon] Auto-installing the Python govon package…');
  console.log(`  → ${pythonCmd} -m pip install govon\n`);

  const result = spawnSync(pythonCmd, ['-m', 'pip', 'install', 'govon'], {
    stdio: 'inherit',
    timeout: 120000,
  });

  if (result.error || result.status !== 0) {
    console.error(
      [
        '',
        '  [govon] Auto-installation failed.',
        '',
        '  Please install manually with:',
        `    ${pythonCmd} -m pip install govon`,
        '',
        '  If you encounter a permission error:',
        `    ${pythonCmd} -m pip install --user govon`,
        '',
      ].join('\n')
    );
    return false;
  }

  console.log('\n  [govon] ✓ Python govon package installed successfully.\n');
  return true;
}

/**
 * Prints the environment check result to stdout and shows guidance if there are issues.
 * Attempts auto-installation if Python govon is not found.
 * @returns {boolean} - true if all conditions are met
 */
function printEnvironmentStatus() {
  const { pythonFound, pythonCmd, pythonVersion, govonInstalled } = checkEnvironment();

  if (!pythonFound) {
    console.error(
      [
        '',
        '  [govon] Python 3.10 or higher is required.',
        '',
        '  Please install Python and try again:',
        '    https://www.python.org/downloads/',
        '',
        '  Or use a package manager:',
        '    macOS:   brew install python@3.12',
        '    Ubuntu:  sudo apt install python3.12',
        '    Windows: winget install Python.Python.3.12',
        '',
      ].join('\n')
    );
    return false;
  }

  if (!govonInstalled) {
    // Attempt auto-installation
    if (!autoInstallGovon(pythonCmd)) {
      return false;
    }
    // Re-verify after installation
    if (!isGovonInstalled()) {
      console.error(
        [
          '',
          '  [govon] govon CLI could not be found even after installation.',
          '',
          '  Please ensure the pip install path is included in your PATH.',
          `  Or run directly: ${pythonCmd} -m govon`,
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
