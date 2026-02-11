#!/usr/bin/env python3
"""
GitHub Subdomain Enumeration Tool
Searches GitHub repositories for subdomain mentions using asyncio for concurrent requests
"""

import asyncio
import logging
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Set, Optional, Dict, Any, Mapping
from dataclasses import dataclass
from collections import deque

import aiohttp
import typer
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Color codes (keeping for legacy/UI if needed, but using logging)
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    NC = "\033[0m"  # No Color


class Settings(BaseSettings):
    """Configuration settings for the application."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    github_token: Optional[str] = None
    github_tokens: Optional[str] = None


@dataclass
class RateLimitInfo:
    """Store rate limit information"""

    limit: int
    remaining: int
    reset: int
    used: int


class TokenRotator:
    """Manages multiple GitHub tokens with rotation"""

    def __init__(self, tokens: List[str], verbose: bool = False):
        self.tokens = deque(tokens)
        self.verbose = verbose
        self.current_token: Optional[str] = None
        self.token_stats: Dict[str, Dict[str, int]] = {
            token: {"requests": 0, "rate_limited": 0} for token in tokens
        }
        self._rotate()

    def _rotate(self) -> None:
        """Rotate to the next token"""
        if not self.tokens:
            raise ValueError("No tokens available")

        # Move current token to end of queue
        if self.current_token:
            self.tokens.append(self.current_token)

        # Get next token
        self.current_token = self.tokens.popleft()

        if self.verbose:
            token_preview = f"{self.current_token[:7]}...{self.current_token[-4:]}"
            logger.info(f"Switched to token: {token_preview}")

    def get_current_token(self) -> str:
        """Get the current active token"""
        if not self.current_token:
            raise ValueError("No token currently active")
        return self.current_token

    def mark_rate_limited(self) -> None:
        """Mark current token as rate limited and rotate"""
        if self.current_token:
            self.token_stats[self.current_token]["rate_limited"] += 1
        logger.warning("Token rate limited. Rotating to next token...")
        self._rotate()

    def increment_request_count(self) -> None:
        """Increment request count for current token"""
        if self.current_token:
            self.token_stats[self.current_token]["requests"] += 1

    def print_stats(self) -> None:
        """Print usage statistics for all tokens"""
        logger.info("Token Usage Statistics:")
        for i, (token, stats) in enumerate(self.token_stats.items(), 1):
            token_preview = f"{token[:7]}...{token[-4:]}"
            logger.info(
                f"  Token {i} ({token_preview}): "
                f"{stats['requests']} requests, "
                f"{stats['rate_limited']} rate limits hit"
            )


class GitHubSubdomainScanner:
    """Main scanner class using asyncio"""

    def __init__(
        self,
        token_rotator: TokenRotator,
        verbose: bool = False,
        max_pages: int = 5,
        concurrent_limit: int = 5,
    ):
        self.token_rotator = token_rotator
        self.verbose = verbose
        self.max_pages = max_pages
        self.concurrent_limit = concurrent_limit
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        self.total_subdomains: Set[str] = set()

    async def __aenter__(self) -> "GitHubSubdomainScanner":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.session:
            await self.session.close()

    def get_headers(self) -> dict[str, str]:
        """Get headers with current token"""
        return {
            "Authorization": f"token {self.token_rotator.get_current_token()}",
            "Accept": "application/vnd.github.v3+json",
        }

    async def validate_token(self) -> bool:
        """Validate the current GitHub token"""
        url = "https://api.github.com/rate_limit"

        if not self.session:
            return False

        try:
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    rate = data["rate"]
                    reset_time = datetime.fromtimestamp(rate["reset"])

                    logger.info(
                        f"Token valid. Rate limit: {rate['remaining']}/{rate['limit']} remaining"
                    )
                    if self.verbose:
                        logger.info(f"Rate limit resets at: {reset_time}")
                    return True
                else:
                    logger.error(f"Invalid token or API error (HTTP {response.status})")
                    return False
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False

    async def handle_rate_limit(
        self, headers: Mapping[str, Any], retry_count: int = 0
    ) -> bool:
        """Handle rate limiting with exponential backoff"""
        remaining = int(headers.get("x-ratelimit-remaining", 1))
        reset = int(headers.get("x-ratelimit-reset", 0))
        retry_after = headers.get("retry-after")

        if self.verbose and remaining is not None:
            logger.info(f"Rate limit remaining: {remaining}")

        # Check if we need to wait
        if remaining == 0 or retry_after:
            if retry_after:
                wait_time = int(retry_after)
                logger.warning(f"Secondary rate limit. Waiting {wait_time}s...")
            else:
                current_time = int(datetime.now().timestamp())
                wait_time = max(reset - current_time + 5, 0)
                logger.warning(
                    f"Primary rate limit. Waiting {wait_time}s until reset..."
                )

            if wait_time > 0:
                # Try to rotate to another token instead of waiting
                if len(self.token_rotator.tokens) > 0:
                    self.token_rotator.mark_rate_limited()
                    # Check if the new token also has rate limit
                    rate_check_url = "https://api.github.com/rate_limit"
                    try:
                        if self.session:
                            async with self.session.get(
                                rate_check_url, headers=self.get_headers()
                            ) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    new_remaining = data["rate"]["remaining"]
                                    if new_remaining > 10:
                                        logger.info(
                                            f"Switched to token with {new_remaining} requests remaining"
                                        )
                                        return True  # Retry with new token immediately
                                    else:
                                        logger.warning(
                                            f"New token also low on requests ({new_remaining} remaining)"
                                        )
                    except Exception:
                        pass

                # If all tokens are exhausted, we have to wait
                logger.warning(f"All tokens exhausted. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)

            return True  # Retry after wait

        # Proactively slow down if approaching limit
        if remaining < 10:
            logger.warning(
                f"Approaching rate limit ({remaining} remaining). Slowing down..."
            )
            await asyncio.sleep(2)

        return False  # No retry needed

    async def make_request(
        self, url: str, max_retries: int = 5
    ) -> Optional[dict[str, Any]]:
        """Make an API request with retry logic"""
        retry_count = 0

        if not self.session:
            return None

        while retry_count < max_retries:
            try:
                async with self.semaphore:
                    self.token_rotator.increment_request_count()

                    async with self.session.get(
                        url, headers=self.get_headers()
                    ) as response:
                        # Handle rate limiting
                        if response.status in [403, 429]:
                            should_retry = await self.handle_rate_limit(
                                response.headers, retry_count
                            )
                            if should_retry:
                                retry_count += 1
                                await asyncio.sleep(
                                    2**retry_count
                                )  # Exponential backoff
                                continue
                            else:
                                return None

                        if response.status == 200:
                            data: Dict[str, Any] = await response.json()
                            return data
                        else:
                            if self.verbose:
                                logger.warning(f"HTTP {response.status} for {url}")
                            return None

            except asyncio.TimeoutError:
                if self.verbose:
                    logger.warning(f"Timeout for {url}")
                retry_count += 1
                await asyncio.sleep(2**retry_count)

            except Exception as e:
                if self.verbose:
                    logger.error(f"Error: {e}")
                retry_count += 1
                await asyncio.sleep(2**retry_count)

        logger.error(f"Max retries reached for {url}")
        return None

    async def fetch_raw_content(self, url: str) -> str:
        """Fetch raw content from a URL"""
        if not self.session:
            return ""
        try:
            async with self.semaphore:
                timeout = aiohttp.ClientTimeout(total=10)
                async with self.session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.text()
        except Exception:
            pass
        return ""

    def extract_subdomains(self, content: str, domain: str) -> Set[str]:
        """Extract subdomains from content"""
        pattern = rf"[a-zA-Z0-9._-]+\.{re.escape(domain)}"
        matches = re.findall(pattern, content)
        return set(matches)

    async def search_github_code(self, domain: str) -> List[str]:
        """Search GitHub code for domain mentions"""
        urls = []

        for page in range(1, self.max_pages + 1):
            if self.verbose:
                logger.info(f"Fetching GitHub page {page}/{self.max_pages}...")

            search_url = f"https://api.github.com/search/code?q=%22.{domain}%22+in:file&page={page}&per_page=100"
            data = await self.make_request(search_url)

            if not data or "items" not in data:
                if self.verbose:
                    logger.warning(f"No items on page {page}. Stopping pagination.")
                break

            items = data["items"]
            if not items:
                if self.verbose:
                    logger.info("No more results. Stopping pagination.")
                break

            for item in items:
                html_url = item.get("html_url", "")
                if html_url:
                    # Convert to raw URL
                    raw_url = html_url.replace(
                        "github.com", "raw.githubusercontent.com"
                    ).replace("/blob/", "/")
                    urls.append(raw_url)

            # Small delay between pages
            if page < self.max_pages:
                await asyncio.sleep(1)

        return urls

    async def scan_domain(self, domain: str, save_individual: bool = True) -> Set[str]:
        """Scan a single domain"""
        logger.info(f"Scanning {domain}")

        # Check if previous results exist
        output_file = f"{domain}.txt"
        previous_subdomains: Set[str] = set()

        if os.path.exists(output_file):
            if self.verbose:
                logger.info(
                    "Found existing results file. Will merge with new findings."
                )
            try:
                with open(output_file, "r") as f:
                    previous_subdomains = {line.strip() for line in f if line.strip()}
                if self.verbose:
                    logger.info(
                        f"Loaded {len(previous_subdomains)} previous subdomains"
                    )
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not read previous results: {e}")

        # Search GitHub for files containing the domain
        urls = await self.search_github_code(domain)

        if not urls:
            logger.warning(f"No GitHub URLs found for {domain}")
            # If we have previous results, keep them
            if previous_subdomains:
                logger.info(f"Keeping {len(previous_subdomains)} previous subdomains")
                return previous_subdomains
            return set()

        if self.verbose:
            logger.info(f"Found {len(urls)} GitHub URLs")

        # Fetch raw content concurrently
        if self.verbose:
            logger.info("Fetching raw content & extracting subdomains...")

        tasks = [self.fetch_raw_content(url) for url in urls]
        contents = await asyncio.gather(*tasks)

        # Extract subdomains from all content
        new_subdomains: Set[str] = set()
        for content in contents:
            if content:
                found = self.extract_subdomains(content, domain)
                new_subdomains.update(found)

        # Merge with previous results
        all_subdomains = previous_subdomains | new_subdomains

        if all_subdomains:
            new_count = len(new_subdomains - previous_subdomains)

            if previous_subdomains:
                logger.info(
                    f"{domain} => Found {len(all_subdomains)} total subdomains "
                    f"({len(previous_subdomains)} previous + {new_count} new)"
                )
            else:
                logger.info(f"{domain} => Found {len(all_subdomains)} subdomains")

            # Save to temporary file first (atomic write)
            if save_individual:
                temp_file = f"{output_file}.tmp"
                try:
                    with open(temp_file, "w") as f:
                        for subdomain in sorted(all_subdomains):
                            f.write(f"{subdomain}\n")

                    # Only replace original if write succeeded
                    os.replace(temp_file, output_file)

                    if new_count > 0:
                        logger.info(
                            f"Updated results saved to: {output_file} (+{new_count} new)"
                        )
                    else:
                        logger.info("No new subdomains found. File unchanged.")

                except Exception as e:
                    logger.error(f"Error saving results: {e}")
                    # Clean up temp file if it exists
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        else:
            logger.warning(f"{domain} => No subdomains extracted")

        if self.verbose:
            logger.info(f"Done with {domain}")

        return all_subdomains

    async def scan_domains(
        self, domains: List[str], output_file: Optional[str] = None
    ) -> Set[str]:
        """Scan multiple domains"""
        logger.info(f"Processing {len(domains)} domain(s)")
        logger.info(
            "Individual results will be saved as <domain>.txt in current directory"
        )

        all_subdomains: Set[str] = set()

        for domain in domains:
            subdomains = await self.scan_domain(domain, save_individual=True)
            all_subdomains.update(subdomains)
            self.total_subdomains.update(subdomains)

        # Write all results to combined file if specified
        if output_file and all_subdomains:
            with open(output_file, "w") as f:
                for subdomain in sorted(all_subdomains):
                    f.write(f"{subdomain}\n")

            logger.info(f"Combined results also saved to: {output_file}")

        logger.info(
            f"Total unique subdomains across all domains: {len(all_subdomains)}"
        )

        return all_subdomains


def load_env_tokens() -> List[str]:
    """Load tokens using Pydantic Settings"""
    settings = Settings()
    tokens: List[str] = []

    if settings.github_token:
        tokens.append(settings.github_token)

    if settings.github_tokens:
        tokens.extend(
            [t.strip() for t in settings.github_tokens.split(",") if t.strip()]
        )

    return tokens


def load_tokens(
    token_arg: Optional[List[str]], token_file_arg: Optional[str]
) -> List[str]:
    """Load tokens from various sources"""
    tokens: List[str] = []

    # Priority 1: Token file
    if token_file_arg:
        token_file = Path(token_file_arg)
        if not token_file.exists():
            logger.error(f"Token file '{token_file_arg}' not found")
            raise typer.Exit(code=1)

        with open(token_file, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    tokens.append(line)

        if tokens:
            logger.info(f"Loaded {len(tokens)} token(s) from {token_file_arg}")

    # Priority 2: Command line arguments
    if not tokens and token_arg:
        tokens.extend(token_arg)

    # Priority 3: Environment variables & .env
    if not tokens:
        tokens = load_env_tokens()

    return tokens


def load_domains(
    domain_arg: Optional[str], domain_list_arg: Optional[str]
) -> List[str]:
    """Load domains from file or argument"""
    domains: List[str] = []

    if domain_arg:
        domains.append(domain_arg)

    if domain_list_arg:
        list_path = Path(domain_list_arg)
        if not list_path.exists():
            logger.error(f"File '{domain_list_arg}' not found")
            raise typer.Exit(code=1)

        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    domains.append(line)

    return domains


app = typer.Typer(
    help="GitHub Subdomain Enumeration Tool using AsyncIO",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("version")
def version():
    """Show version"""
    typer.echo("1.0.0")


@app.command("scan")
def scan(
    domain: Optional[str] = typer.Option(
        None, "--domain", "-d", help="Single domain to scan"
    ),
    domain_list: Optional[str] = typer.Option(
        None, "--domain-list", "-l", help="File containing domains (one per line)"
    ),
    token: Optional[List[str]] = typer.Option(
        None,
        "--token",
        "-t",
        help="GitHub token (can be used multiple times for rotation)",
    ),
    token_file: Optional[str] = typer.Option(
        None, "--token-file", "-tf", help="File containing GitHub tokens (one per line)"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Combined output file for all results"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    max_pages: int = typer.Option(
        5, "--max-pages", "-m", help="Maximum pages to fetch per domain"
    ),
    concurrent_limit: int = typer.Option(
        5, "--concurrent-limit", "-c", help="Maximum concurrent requests"
    ),
) -> None:
    """


    Scan GitHub for subdomains.
    """

    if verbose:
        logger.setLevel(logging.DEBUG)

    tokens = load_tokens(token, token_file)

    if not tokens:
        logger.error("GitHub token(s) required")

        raise typer.Exit(code=1)

    domains = load_domains(domain, domain_list)

    if not domains:
        logger.error("Domain (-d) or domain list (-l) required")

        raise typer.Exit(code=1)

    token_rotator = TokenRotator(tokens, verbose=verbose)

    logger.info(f"Loaded {len(tokens)} token(s) for rotation")

    async def run_scanner() -> None:

        async with GitHubSubdomainScanner(
            token_rotator=token_rotator,
            verbose=verbose,
            max_pages=max_pages,
            concurrent_limit=concurrent_limit,
        ) as scanner:
            if not await scanner.validate_token():
                raise typer.Exit(code=1)

            await scanner.scan_domains(domains, output)

            logger.info("Scan complete!")

            if len(tokens) > 1:
                token_rotator.print_stats()

    asyncio.run(run_scanner())


if __name__ == "__main__":
    app()
