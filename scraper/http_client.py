"""
HTTP client functions for fetching HTML content.

Simple functional interface for making HTTP requests with rate limiting
and proper error handling.
"""

import time
import requests
from typing import Optional
from . import config


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
    """
    if delay is None:
        delay = config.REQUEST_DELAY
    
    # Polite rate limiting
    if delay > 0:
        time.sleep(delay)
    
    headers = {
        'User-Agent': config.USER_AGENT
    }
    
    response = requests.get(
        url, 
        timeout=config.REQUEST_TIMEOUT, 
        headers=headers
    )
    response.raise_for_status()
    
    return response.text


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

