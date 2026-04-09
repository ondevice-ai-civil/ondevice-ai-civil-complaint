# govon (npm)

npm wrapper for the [GovOn](https://github.com/umyunsang/GovOn) CLI.

GovOn is a shell-first local agent runtime for administrative support and civil complaint processing workflows.

## Requirements

- **Node.js** 18 or later
- **Python** 3.10 or later
- **pip** (Python package manager)

## Installation

```bash
npm install -g govon
```

After installation the Python environment is automatically verified.  
If the `govon` Python package is not installed, install it with:

```bash
pip install govon
```

## Usage

```bash
govon --help
```

## How It Works

This package is a thin wrapper around the Python CLI (`govon`).

1. When the `govon` command is run, it checks whether Python 3.10+ is installed.
2. It verifies that the `govon` CLI (`pip install govon`) is installed.
3. If all conditions are met, execution is delegated to the Python CLI via `child_process.spawn`.
4. If conditions are not met, a clear installation guide message is printed and the process exits.

> Automatic Python installation is not supported due to security and permission concerns.

## License

MIT
