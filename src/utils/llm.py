"""
LLM Client — Groq API wrapper with automatic retry on rate limits.

Groq's free tier has a 6000 tokens/min limit. This wrapper
automatically waits and retries when rate limits are hit.
"""

import time
import re

from groq import Groq


_client = None


def get_client() -> Groq:
    """Get a shared Groq client instance."""
    global _client
    if _client is None:
        _client = Groq()
    return _client


def chat(messages: list[dict], model: str = "llama-3.1-8b-instant", temperature: float = 0.3, max_retries: int = 5) -> str:
    """Call Groq chat API with automatic retry on rate limits.

    Args:
        messages: List of message dicts (role + content)
        model: Groq model to use
        temperature: Sampling temperature
        max_retries: Max retry attempts on rate limit

    Returns:
        The assistant's response text
    """
    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                # Extract wait time from error message
                match = re.search(r'(\d+\.?\d*)\s*s', error_str)
                wait_time = float(match.group(1)) if match else 20
                wait_time = max(wait_time, 5)
                print(f"   ⏳ Rate limited. Waiting {wait_time:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed after {max_retries} retries due to rate limits")
