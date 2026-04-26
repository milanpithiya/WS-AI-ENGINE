"""
Token Counter — Estimates token count for text.
Uses tiktoken if available, otherwise falls back to simple estimation.
"""

from typing import Optional
from loguru import logger

_encoder = None
_use_tiktoken = False


def _init_encoder():
    """Try to load tiktoken, fall back to simple counting."""
    global _encoder, _use_tiktoken
    try:
        import tiktoken
        _encoder = tiktoken.get_encoding("cl100k_base")
        _use_tiktoken = True
        logger.debug("Using tiktoken for token counting")
    except (ImportError, Exception):
        _use_tiktoken = False
        logger.debug("tiktoken not available, using estimation (1 token ~ 4 chars)")


def count_tokens(text: str) -> int:
    """
    Count tokens in text.
    Uses tiktoken if available, otherwise estimates at ~4 chars per token.
    """
    global _encoder, _use_tiktoken

    if _encoder is None and not _use_tiktoken:
        _init_encoder()

    if _use_tiktoken and _encoder:
        return len(_encoder.encode(text))

    # Fallback: rough estimate
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within max_tokens.
    Returns truncated text with a note if truncation occurred.
    """
    current = count_tokens(text)
    if current <= max_tokens:
        return text

    # Estimate characters to keep
    ratio = max_tokens / current
    keep_chars = int(len(text) * ratio * 0.95)  # 5% margin

    truncated = text[:keep_chars]
    truncated += "\n\n[Content truncated to fit token limit]"

    logger.debug("Truncated from {0} to ~{1} tokens", current, max_tokens)
    return truncated
