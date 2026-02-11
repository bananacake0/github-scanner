#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from github_subdomains.main import app

if __name__ == "__main__":
    # If no subcommand is provided (and it's not --help/-h), insert 'scan'
    # to maintain the "no subcommand" interface while allowing Typer's help features.
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "scan",
        "--help",
        "-h",
        "--install-completion",
        "--show-completion",
    ]:
        sys.argv.insert(1, "scan")
    elif len(sys.argv) == 1:
        sys.argv.append("--help")
        sys.argv.insert(1, "scan")

    app()
