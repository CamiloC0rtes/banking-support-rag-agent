"""Holiday MCP tool. Regression: an unpinned `mcp` 2.x removed FastMCP, the server
crashed on start-up and the agent silently lost holiday awareness."""

import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src import agent, mcp_server


class _Resp:
    def json(self):
        return [{"date": "2026-01-19", "name": "Martin Luther King, Jr. Day"}]


def test_server_tool_formats_holidays(monkeypatch):
    monkeypatch.setattr(mcp_server.requests, "get", lambda *a, **k: _Resp())
    text = mcp_server.get_federal_holidays(2026)
    assert text == "2026-01-19: Martin Luther King, Jr. Day"
    assert agent.holidays_valid(text)


async def test_server_starts_and_exposes_tool_over_stdio():
    params = StdioServerParameters(command=sys.executable, args=["src/mcp_server.py"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
    assert "get_federal_holidays" in [t.name for t in tools.tools]


@pytest.mark.parametrize("payload", [None, "", "[]", "Holiday data currently unavailable."])
def test_error_payloads_are_not_valid_holiday_data(payload):
    assert not agent.holidays_valid(payload)


async def test_failures_are_not_cached(monkeypatch):
    calls = []

    async def flaky(year=None):
        calls.append(year)
        return None if len(calls) == 1 else "2026-01-19: Martin Luther King, Jr. Day"

    monkeypatch.setattr(agent, "_CACHED_HOLIDAYS", None)
    monkeypatch.setattr(agent, "call_mcp_holidays", flaky)
    assert await agent.fetch_holiday_name("2026-01-19") is None
    assert await agent.fetch_holiday_name("2026-01-19") == "Martin Luther King, Jr. Day"
    assert await agent.fetch_holiday_name("2026-01-19") == "Martin Luther King, Jr. Day"
    assert calls == [2026, 2026]  # retried once after the failure, then cached
    assert agent.mcp_ready()
