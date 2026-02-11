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
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Basic Scan

Scan a single domain:

```bash
./github-subdomains scan -d example.com -t YOUR_GITHUB_TOKEN
```

### Multiple Domains

Scan domains from a list:

```bash
./github-subdomains scan -l domains.txt -t YOUR_GITHUB_TOKEN
```

### Token Rotation

Provide multiple tokens via flags or a file:

```bash
./github-subdomains scan -d example.com -t TOKEN1 -t TOKEN2 -tf tokens.txt
```

### Authentication Options

Tokens can be provided in several ways (in order of precedence):
1. `--token-file` / `-tf` flag (one token per line)
2. `--token` / `-t` flag (can be used multiple times)
3. `GITHUB_TOKEN` environment variable
4. `GITHUB_TOKENS` environment variable (comma-separated)
5. `.env` file containing `GITHUB_TOKEN` or `GITHUB_TOKENS`

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
uv pip install pytest
uv run pytest
```