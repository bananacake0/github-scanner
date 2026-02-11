#!/usr/bin/env python3
import sys
from pathlib import Path
import typer

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from github_subdomains.main import scan

if __name__ == "__main__":
    typer.run(scan)