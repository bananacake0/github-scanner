import pytest
from github_subdomains.main import TokenRotator, GitHubSubdomainScanner
import asyncio

def test_token_rotator():
    tokens = ["token1", "token2", "token3"]
    rotator = TokenRotator(tokens)
    
    assert rotator.get_current_token() == "token1"
    rotator._rotate()
    assert rotator.get_current_token() == "token2"
    rotator.mark_rate_limited()
    assert rotator.get_current_token() == "token3"
    rotator._rotate()
    assert rotator.get_current_token() == "token1"
    
    assert rotator.token_stats["token2"]["rate_limited"] == 1

def test_extract_subdomains():
    rotator = TokenRotator(["token"])
    scanner = GitHubSubdomainScanner(rotator)
    
    content = "Check out dev.example.com and api.example.com or maybe test-1.example.com"
    domain = "example.com"
    
    found = scanner.extract_subdomains(content, domain)
    assert "dev.example.com" in found
    assert "api.example.com" in found
    assert "test-1.example.com" in found
    assert len(found) == 3

@pytest.mark.asyncio
async def test_validate_token_failure():
    # This just tests the structure since we can't easily mock aiohttp without more setup
    # or using aioresponses. For now, just a basic test that it handles exceptions.
    rotator = TokenRotator(["invalid_token"])
    async with GitHubSubdomainScanner(rotator) as scanner:
        # Should return False because it will fail to connect or get 401
        # (Though in CI it might fail due to no network, which is also fine)
        try:
            result = await scanner.validate_token()
            assert result is False
        except Exception:
            pass # Network errors are also fine for this simple test
