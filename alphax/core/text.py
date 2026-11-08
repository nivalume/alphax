"""Bounded excerpts of long texts for prompts and logs."""

from __future__ import annotations


class Excerpt:
    """Cut long texts so they fit a prompt while keeping the informative parts."""

    @staticmethod
    def middle(text: str, limit: int) -> str:
        """Keep the head and the tail, drop the middle."""
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return f"{text[:half]}\n... [{len(text) - 2 * half} characters omitted] ...\n{text[-half:]}"

    @staticmethod
    def tail(text: str, limit: int) -> str:
        """Keep the end, where tracebacks and summaries are."""
        if len(text) <= limit:
            return text
        return f"... [{len(text) - limit} characters omitted] ...\n{text[-limit:]}"
