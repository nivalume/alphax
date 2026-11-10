"""Search alphax's documentation: every public docstring, plus Markdown files.

::

    alphax.docs.search("trailing stop")           # [Hit(name, score, snippet), ...]
    alphax.docs.search("cross-sectional rank", limit=3)
    index = alphax.docs.Index(extra=["README.md"])   # add a checkout's Markdown

The index is built from the installed package on first use: the docstrings
of every public module, class, function and method under ``alphax``, each
split into paragraphs. Markdown files and directories given as ``extra``
are split at their headings. Ranking is BM25 over words, with names and
headings counted twice. Nothing leaves the machine.
"""

from __future__ import annotations

import inspect
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Hit", "Index", "search"]

_WORD = re.compile(r"[a-z0-9_]+|[一-鿿]")


def _words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        out.append(w)
        if "_" in w:
            out.extend(p for p in w.split("_") if p)
    return out


@dataclass(frozen=True)
class Hit:
    """One matching passage: where it is, its score and its text."""

    name: str
    score: float
    text: str

    @property
    def snippet(self) -> str:
        return self.text if len(self.text) <= 300 else self.text[:297] + "..."


@dataclass(frozen=True)
class _Doc:
    name: str
    title: str
    text: str


def _public(module: Any, prefix: str, seen: set[int]) -> Iterable[tuple[str, Any]]:
    names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
    for n in names:
        obj = getattr(module, n, None)
        if obj is None or id(obj) in seen or inspect.ismodule(obj):
            continue
        # Skip what a module only imported (NumPy functions and the like).
        if not str(getattr(obj, "__module__", "alphax") or "alphax").startswith("alphax"):
            continue
        seen.add(id(obj))
        yield f"{prefix}.{n}", obj
        if inspect.isclass(obj):
            for m, member in vars(obj).items():
                if not m.startswith("_") and (inspect.isfunction(member) or isinstance(member, property)
                                              or callable(member)):
                    yield f"{prefix}.{n}.{m}", member


def _docstrings() -> list[_Doc]:
    import alphax

    seen: set[int] = set()
    out: list[_Doc] = []
    modules = [("alphax", alphax)] + [
        (f"alphax.{n}", m) for n, m in sorted(vars(alphax).items())
        if inspect.ismodule(m) and not n.startswith("_") and m.__name__.startswith("alphax.")
    ]
    for prefix, module in modules:
        if module.__doc__:
            out.extend(_paragraphs(prefix, prefix, module.__doc__))
        for name, obj in _public(module, prefix, seen):
            doc = inspect.getdoc(obj)
            if doc:
                out.extend(_paragraphs(name, name.rsplit(".", 1)[-1], doc))
    return out


def _paragraphs(name: str, title: str, text: str) -> list[_Doc]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", inspect.cleandoc(text)) if p.strip()]
    # Keep a code example with the paragraph that introduces it.
    merged: list[str] = []
    for p in parts:
        if merged and (p.startswith((" ", ">>>")) or merged[-1].endswith("::")):
            merged[-1] += "\n\n" + p
        else:
            merged.append(p)
    return [_Doc(name, title, p) for p in merged]


def _markdown(path: Path) -> list[_Doc]:
    out = []
    files = sorted(path.rglob("*.md")) if path.is_dir() else [path]
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        heading, buf = f.name, []
        for line in text.splitlines():
            if line.startswith("#"):
                if "".join(buf).strip():
                    out.append(_Doc(f"{f}#{heading}", heading, "\n".join(buf).strip()))
                heading, buf = line.lstrip("#").strip(), []
            else:
                buf.append(line)
        if "".join(buf).strip():
            out.append(_Doc(f"{f}#{heading}", heading, "\n".join(buf).strip()))
    return out


class Index:
    """A BM25 index over the docstrings and any ``extra`` Markdown."""

    def __init__(self, extra: Iterable[str | Path] = (), docstrings: bool = True):
        self.docs: list[_Doc] = _docstrings() if docstrings else []
        for p in extra:
            self.docs.extend(_markdown(Path(p)))
        self._terms = [Counter(_words(d.text) + 2 * _words(d.title + " " + d.name)) for d in self.docs]
        self._lens = [sum(t.values()) for t in self._terms]
        self._avg = sum(self._lens) / max(len(self._lens), 1)
        df: Counter[str] = Counter()
        for t in self._terms:
            df.update(t.keys())
        n = len(self.docs)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, limit: int = 10) -> list[Hit]:
        """The ``limit`` best passages for ``query``."""
        q = _words(query)
        k1, b = 1.2, 0.75
        scored = []
        for i, terms in enumerate(self._terms):
            s = 0.0
            for w in q:
                f = terms.get(w, 0)
                if f:
                    s += self._idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * self._lens[i] / self._avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Hit(self.docs[i].name, round(s, 4), self.docs[i].text) for s, i in scored[:limit]]


_DEFAULT: Index | None = None


def search(query: str, limit: int = 10) -> list[Hit]:
    """Searches the installed package's docstrings."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Index()
    return _DEFAULT.search(query, limit)
