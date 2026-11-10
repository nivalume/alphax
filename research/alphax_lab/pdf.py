"""Text from a literature PDF, chunked with page numbers for card sourcing."""

from __future__ import annotations

from pathlib import Path
from typing import cast


def chunks(path: str | Path, pages: tuple[int, int] | None = None, max_chars: int = 40_000) -> list[dict]:
    """Page-tagged text chunks; `pages` is an inclusive 1-based range (a book chapter, say)."""
    import pymupdf

    out, text, first = [], [], pages[0] if pages else 1
    with pymupdf.open(Path(path).expanduser()) as document:
        for number in range(1, document.page_count + 1):
            page = document.load_page(number - 1)
            if pages and not pages[0] <= number <= pages[1]:
                continue
            body = cast(str, page.get_text("text")).strip()
            if not body:
                continue
            if text and sum(len(t) for t in text) + len(body) > max_chars:
                out.append({"pages": [first, number - 1], "text": "\n".join(text)})
                text, first = [], number
            text.append(f"[page {number}]\n{body}")
    if text:
        out.append({"pages": [first, number], "text": "\n".join(text)})
    if not out:
        raise ValueError(f"{path} has no extractable text (scanned PDF?)")
    return out
