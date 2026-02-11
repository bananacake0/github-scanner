# GitHub Subdomain Enumeration Tool

A robust, asynchronous CLI tool for searching GitHub repositories for subdomain mentions.

## Features

- **Asynchronous Execution:** Built with `asyncio` and `aiohttp` for high-performance scanning.
- **Token Rotation:** Automatically rotates through multiple GitHub Personal Access Tokens (PATs) to maximize rate limits.
- **Rate Limit Handling:** Intelligent handling of primary and secondary GitHub API rate limits.
- **Result Merging:** Automatically merges new findings with existing results files.
- **Flexible Configuration:** Supports environment variables, `.env` files, and command-line arguments.

## Installation

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

```bash
git clone https://github.com/yourusername/github-scanner.git
cd github-scanner
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Basic Scan

Scan a single domain:

```bash
uv run python3 github-subdomains.py -d example.com -t YOUR_GITHUB_TOKEN
```

### Shell Completion

This tool supports tab-completion for flags and options to make the CLI easier to use.

- **`--install-completion`**: Automatically detects your shell (Bash, Zsh, or Fish) and adds a completion script to your shell's configuration file (e.g., `.zshrc`). After running this and restarting your terminal, you can press `Tab` to autocomplete flags like `-d` or `-tf`.
- **`--show-completion`**: Displays the completion script in the terminal without installing it. This is useful for manual inspection or custom setups.

### Authentication Options

Tokens can be provided in several ways (in order of precedence):
1. `--token-file` / `-tf` flag (one token per line)
2. `--token` / `-t` flag (can be used multiple times)
3. `GITHUB_TOKEN` environment variable
4. `GITHUB_TOKENS` environment variable (comma-separated)
5. `.env` file containing `GITHUB_TOKEN` or `GITHUB_TOKENS`

## Repository & Ignored Files

The project includes a `.gitignore` file to ensure that temporary, private, or environment-specific files are not committed to the repository:

- **`.venv/`**: Keeps the Python virtual environment local.
- **`*.txt` / `*.tmp`**: Prevents scan results (like `example.com.txt`) from being accidentally committed.
- **`.env`**: Protects your private GitHub tokens and configuration.
- **`todo.md`**: A local-only file used for tracking development progress.
- **`__pycache__/`**: Standard Python bytecode cache.

## Development

This project uses the `src/` layout and `uv` for environment management.

### Linting & Formatting

```bash
uv pip install ruff mypy
uv run ruff check .
uv run ruff format .
uv run mypy .
```

### Testing

```bash
uv pip install pytest pytest-asyncio
PYTHONPATH=src uv run pytest
```
