"""Documentation search and the MCP server."""

import io
import json
from pathlib import Path

import pytest

import alphax
from alphax import mcp

ROOT = Path(__file__).resolve().parents[3]


def test_doc_search_finds_the_right_function():
    hits = alphax.docs.search("trailing stop loss", limit=10)
    assert hits and all(h.score > 0 for h in hits)
    assert any("stop" in h.name for h in hits[:3]) and not any(".np." in h.name for h in hits)
    assert alphax.docs.search("cross sectional rank percentile", limit=20)
    top = alphax.docs.search("signal_delay_test", limit=3)
    assert top[0].name.endswith("signal_delay_test")
    md = alphax.docs.Index(extra=[ROOT / "README.md"], docstrings=False)
    assert md.search("WorldQuant 101")[0].name.startswith(str(ROOT / "README.md"))


def _rpc(server, *messages):
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n"), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _call(server, tool, **args):
    (reply,) = _rpc(server, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": tool, "arguments": args}})
    result = reply["result"]
    return result["isError"], result["content"][0]["text"]


def test_mcp_handshake_and_tool_list():
    s = mcp.Server(ROOT)
    init, tools = _rpc(
        s,
        {"jsonrpc": "2.0", "id": 0, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert init["result"]["serverInfo"]["name"] == "alphax"
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert {t["name"] for t in tools["result"]["tools"]} == {"search_docs", "api_reference", "read_source",
                                                             "run_backtest"}
    (bad,) = _rpc(s, {"jsonrpc": "2.0", "id": 9, "method": "nope"})
    assert bad["error"]["code"] == -32601


def test_mcp_tools():
    s = mcp.Server(ROOT)
    err, text = _call(s, "search_docs", query="row_offset resume", limit=3)
    assert not err and "##" in text
    err, text = _call(s, "api_reference", name="indicators.rsi")
    assert not err and text.startswith("rsi(")
    err, text = _call(s, "read_source", name="allocation.equal_weight")
    assert not err and "def equal_weight" in text
    err, text = _call(s, "read_source", path="crates/alphax-expr/src/cross.rs", lines=5)
    assert not err and "more lines" in text
    err, text = _call(s, "run_backtest", entries="$close > Mean($close, 20)", exits="$close < Mean($close, 20)",
                      n_rows=300, n_cols=2, options={"size": 1.0, "init_cash": 1e4})
    assert not err
    out = json.loads(text)
    assert out["columns"] == 2 and out["orders"] > 0 and "Total Return [%]" in out["stats"]["column 0"]


@pytest.mark.parametrize("path", ["research/alphax_lab/cli.py", "../../etc/passwd", "pyproject.toml",
                                  "/etc/hostname"])
def test_mcp_cannot_read_research_or_outside(path):
    err, text = _call(mcp.Server(ROOT), "read_source", path=path)
    assert err and ("not readable" in text or "outside" in text)


def test_mcp_refuses_large_or_file_inputs():
    s = mcp.Server(None)
    err, text = _call(s, "run_backtest", entries="1", exits="0", n_rows=100000, n_cols=10)
    assert err and "200000" in text
    err, text = _call(s, "read_source", path="crates/x.rs")
    assert err and "--source" in text
    err, _ = _call(s, "run_backtest", entries="1", exits="0", path="data.csv")
    assert err
