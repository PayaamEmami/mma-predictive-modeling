"""
HTTP client functions for fetching HTML content.

Simple functional interface for making HTTP requests with rate limiting
and proper error handling. Handles ufcstats.com browser-check challenges
via a lightweight proof-of-work handshake (no browser required).
"""

import hashlib
import re
import time
from typing import Optional
from urllib.parse import urlparse

import requests

from . import config

_CHALLENGE_MARKER = "Checking your browser"
_CHALLENGE_NONCE_RE = re.compile(r'nonce="([a-f0-9]+)"')
_CHALLENGE_DIFFICULTY_RE = re.compile(r"new Array\((\d+)\+1\)\.join\('0'\)")


def _request_headers() -> dict[str, str]:
    return {"User-Agent": config.USER_AGENT}


def _is_challenge_page(html: str) -> bool:
    return _CHALLENGE_MARKER in html


def _parse_challenge(html: str) -> tuple[str, int]:
    nonce_match = _CHALLENGE_NONCE_RE.search(html)
    difficulty_match = _CHALLENGE_DIFFICULTY_RE.search(html)
    if not nonce_match:
        raise ValueError("Challenge page missing nonce")
    difficulty = int(difficulty_match.group(1)) if difficulty_match else 2
    return nonce_match.group(1), difficulty


def _solve_proof_of_work(nonce: str, difficulty: int) -> int:
    target = "0" * difficulty
    n = 0
    while True:
        digest = hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest()
        if digest.startswith(target):
            return n
        n += 1


def _challenge_check_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/__c"


def _pass_challenge(session: requests.Session, url: str, challenge_html: str) -> None:
    nonce, difficulty = _parse_challenge(challenge_html)
    solution = _solve_proof_of_work(nonce, difficulty)
    check_url = _challenge_check_url(url)
    response = session.post(
        check_url,
        data={"nonce": nonce, "n": str(solution)},
        timeout=config.REQUEST_TIMEOUT,
        headers=_request_headers(),
    )
    response.raise_for_status()


def fetch_html(url: str, delay: Optional[float] = None) -> str:
    """
    Fetch HTML content from a URL with rate limiting.

    Args:
        url: URL to fetch
        delay: Optional delay in seconds before request (defaults to config.REQUEST_DELAY)

    Returns:
        HTML content as string

    Raises:
        requests.RequestException: If the request fails
        ValueError: If a challenge page cannot be cleared
    """
    if delay is None:
        delay = config.REQUEST_DELAY

    if delay > 0:
        time.sleep(delay)

    with requests.Session() as session:
        session.headers.update(_request_headers())
        response = session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text

        if _is_challenge_page(html):
            _pass_challenge(session, url, html)
            response = session.get(url, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            html = response.text
            if _is_challenge_page(html):
                raise ValueError(
                    "Failed to pass ufcstats.com browser check after proof-of-work"
                )

        return html


def fetch_multiple_pages(urls: list[str], delay: Optional[float] = None) -> list[str]:
    """
    Fetch multiple pages sequentially with rate limiting.

    Args:
        urls: List of URLs to fetch
        delay: Optional delay between requests

    Returns:
        List of HTML content strings

    Raises:
        requests.RequestException: If any request fails
    """
    return [fetch_html(url, delay) for url in urls]
