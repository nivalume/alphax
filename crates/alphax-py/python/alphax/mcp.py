"""alphax as an MCP server: look up the API, read the source, run small backtests.

Run it over stdio (the transport every MCP client supports)::

    python -m alphax.mcp [--source /path/to/alphax/checkout]
    alphax mcp [--source ...]

and register that command with the client (for Claude Code:
``claude mcp add alphax -- python -m alphax.mcp --source .``).

Tools:

* ``search_docs(query, limit)``: :func:`alphax.docs.search` over docstrings,
  and the checkout's ``README.md`` when ``--source`` is given.
* ``api_reference(name)``: signature and docstring of ``alphax.<name>``.
* ``read_source(name | path)``: the Python source of ``alphax.<name>``, or a
  file of the checkout under ``crates/``, ``examples/``,
  ``skills/``, ``benchmarks/``, ``tests/``, ``README.md`` or ``AGENTS.md``.
* ``run_backtest(...)``: ``from_signals`` on synthetic prices or on small
  inline arrays (at most 200,000 cells), signals as factor expressions;
  returns the statistics.

What it deliberately cannot reach: ``research/`` (``alphax_lab``'s data and
ledger), any other path, and any file as backtest input. A model connected
here can learn and try the engine without seeing research results.

The server speaks JSON-RPC 2.0, one message per line, MCP protocol version
``2025-06-18``; it needs no SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np

__all__ = ["Server", "main"]

PROTOCOL = "2025-06-18"
MAX_CELLS = 200_000
_ALLOWED = ("crates", "examples", "skills", "benchmarks", "tests")
_ALLOWED_FILES = ("README.md", "AGENTS.md")


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {
        "name": "search_docs",
        "description": "Search alphax's documentation (docstrings, and the checkout's README.md when available).",
        "inputSchema": _schema({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                               ["query"]),
    },
    {
        "name": "api_reference",
        "description": "Signature, docstring and members of alphax.<name>, e.g. 'from_signals' or 'indicators.rsi'.",
        "inputSchema": _schema({"name": {"type": "string"}}, ["name"]),
    },
    {
        "name": "read_source",
        "description": "Python source of alphax.<name>, or a file of the alphax checkout by relative path "
                       "(crates/, examples/, skills/, benchmarks/, tests/, README.md, AGENTS.md).",
        "inputSchema": _schema({"name": {"type": "string"}, "path": {"type": "string"},
                                "start": {"type": "integer", "minimum": 1}, "lines": {"type": "integer", "minimum": 1}},
                               []),
    },
    {
        "name": "run_backtest",
        "description": "alphax.from_signals on synthetic GBM prices (n_rows, n_cols, seed) or inline close "
                       "(list of rows), with entries/exits as factor expressions over $close, $high, $low; "
                       "options are from_signals keyword arguments, plus warmup (default true: no signal "
                       "before an expression's windows have filled). Returns statistics per column.",
        "inputSchema": _schema({
            "entries": {"type": "string"}, "exits": {"type": "string"},
            "close": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "n_rows": {"type": "integer", "minimum": 2}, "n_cols": {"type": "integer", "minimum": 1},
            "seed": {"type": "integer"}, "options": {"type": "object"},
        }, ["entries", "exits"]),
    },
]


class ToolError(Exception):
    """A tool call that cannot be served; reported to the client as an error result."""


class Server:
    """The tools and the JSON-RPC loop."""

    def __init__(self, source: str | Path | None = None):
        self.source = Path(source).resolve() if source else None
        self._index = None

    # --- tools -------------------------------------------------------------

    def search_docs(self, query: str, limit: int = 5) -> str:
        from .docs import Index

        if self._index is None:
            extra = []
            if self.source is not None:
                extra = [p for p in (self.source / "README.md",) if p.exists()]
            self._index = Index(extra=extra)
        hits = self._index.search(query, limit)
        if not hits:
            return "no matches"
        return "\n\n".join(f"## {h.name} (score {h.score})\n{h.snippet}" for h in hits)

    def api_reference(self, name: str) -> str:
        from .tools import help as alphax_help

        try:
            return alphax_help(name, echo=False)
        except AttributeError as e:
            raise ToolError(str(e)) from None

    def read_source(self, name: str | None = None, path: str | None = None, start: int = 1,
                    lines: int = 400) -> str:
        if (name is None) == (path is None):
            raise ToolError("give exactly one of name and path")
        if name is not None:
            import inspect

            from .tools import _resolve

            try:
                obj = _resolve(name)
                text = inspect.getsource(obj)
            except (AttributeError, TypeError, OSError) as e:
                raise ToolError(f"no Python source for {name!r}: {e}") from None
        else:
            text = self._read_checkout(path)
        all_lines = text.splitlines()
        chunk = all_lines[start - 1:start - 1 + lines]
        more = len(all_lines) - (start - 1 + len(chunk))
        body = "\n".join(f"{i:5d}  {line}" for i, line in enumerate(chunk, start))
        return body + (f"\n... {more} more lines; pass start={start + len(chunk)}" if more > 0 else "")

    def _read_checkout(self, path: str) -> str:
        if self.source is None:
            raise ToolError("no checkout: start the server with --source")
        target = (self.source / path).resolve()
        try:
            rel = target.relative_to(self.source)
        except ValueError:
            raise ToolError("path is outside the checkout") from None
        top = rel.parts[0] if rel.parts else ""
        if not (top in _ALLOWED or (len(rel.parts) == 1 and top in _ALLOWED_FILES)):
            raise ToolError(f"{rel} is not readable here; allowed: {', '.join(_ALLOWED + _ALLOWED_FILES)}")
        if not target.is_file():
            raise ToolError(f"{rel} is not a file")
        return target.read_text(encoding="utf-8", errors="replace")

    def run_backtest(self, entries: str, exits: str, close: list | None = None, n_rows: int = 500,
                     n_cols: int = 1, seed: int = 0, options: dict | None = None) -> str:
        import alphax

        if close is not None:
            c = np.asarray(close, dtype=float)
            if c.ndim != 2:
                raise ToolError("close must be a list of rows")
        else:
            if n_rows * n_cols > MAX_CELLS:
                raise ToolError(f"at most {MAX_CELLS} cells")
            c = alphax.synthetic.gbm((n_rows, n_cols), seed=seed)
        if c.size > MAX_CELLS:
            raise ToolError(f"at most {MAX_CELLS} cells")
        high, low, _ = alphax.synthetic.ohlc_from_close(c, seed=seed) if close is None else (c, c, None)
        fields = {"close": np.asfortranarray(c), "high": np.asfortranarray(high), "low": np.asfortranarray(low)}
        options = dict(options or {})
        for bad in ("resume", "callbacks", "pre_group_func", "signal_func", "adjust_func",
                    "post_order_func", "post_segment_func"):
            options.pop(bad, None)
        from .config import _signal

        warmup = bool(options.pop("warmup", True))
        try:
            factors = alphax.FactorSet.compile({"entries": entries, "exits": exits}).with_warmup(warmup)
            sig = factors.evaluate(fields)
            pf = alphax.from_signals(fields["close"], _signal(sig["entries"]), _signal(sig["exits"]),
                                     high=fields["high"], low=fields["low"], **options)
        except (ValueError, TypeError) as e:
            raise ToolError(str(e)) from None
        from .config import _plain

        stats = {f"column {k}": _plain(pf.stats(k)) for k in range(pf.n_cols)} if not pf.cash_sharing else \
            {f"group {g}": _plain(pf.stats(group=g)) for g in range(len(pf.group_lens))}
        return json.dumps({"bars": int(c.shape[0]), "columns": int(c.shape[1]),
                           "orders": int(len(pf.orders()["idx"])), "stats": stats}, indent=1)

    # --- protocol ----------------------------------------------------------

    def handle(self, msg: dict) -> dict | None:
        """The response to one JSON-RPC message, or ``None`` for a notification."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            return None
        try:
            if method == "initialize":
                version = msg.get("params", {}).get("protocolVersion", PROTOCOL)
                result: Any = {
                    "protocolVersion": version if isinstance(version, str) else PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "alphax", "version": _version()},
                    "instructions": "Tools to look up alphax's API and source and to run small backtests "
                                    "on synthetic or inline data.",
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call(msg.get("params", {}))
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        except Exception as e:  # a bug in a tool must not kill the server
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _call(self, params: dict) -> dict:
        name, args = params.get("name"), params.get("arguments") or {}
        tools: dict[str, Callable[..., str]] = {
            "search_docs": self.search_docs, "api_reference": self.api_reference,
            "read_source": self.read_source, "run_backtest": self.run_backtest,
        }
        if name not in tools:
            return {"content": [{"type": "text", "text": f"no tool {name!r}"}], "isError": True}
        try:
            text = tools[name](**args)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        except (ToolError, TypeError) as e:
            return {"content": [{"type": "text", "text": str(e)}], "isError": True}

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        """Reads messages line by line until end of input."""
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                reply: dict | None = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            else:
                reply = self.handle(msg)
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("alphax-trader")
    except Exception:
        return "0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphax mcp", description="alphax's MCP server over stdio")
    parser.add_argument("--source", help="a alphax checkout to search and read")
    args = parser.parse_args(argv)
    Server(args.source).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
