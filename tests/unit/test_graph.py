"""End-to-end graph behavior with the LLM, retriever and MCP tool mocked."""

import pytest
from langchain_core.documents import Document

import src.agent as agent
import src.database as database

TRUSTED_DEVICE = Document(
    page_content="When a user selects 'Remember this device' ... This cookie is valid for exactly 30 days.",
    metadata={"source": "Detailed_Security_Protocol.pdf", "page": 3, "tags": "remember_me"},
)


@pytest.fixture
def calls(monkeypatch):
    """Mock LLM + MCP. Returns a list of system prompts sent to the LLM."""
    prompts = []

    async def fake_complete(system_prompt, history, user_message, temp=0.2, top_p=0.9):
        prompts.append(system_prompt)
        return "FAKE_ANSWER"

    async def fake_holiday(_date):
        return "Martin Luther King, Jr. Day" if _date == "2026-01-19" else None

    monkeypatch.setattr(agent, "complete", fake_complete)
    monkeypatch.setattr(agent, "fetch_holiday_name", fake_holiday)
    return prompts


def mock_retrieval(monkeypatch, results):
    async def fake_search(query, k=4):
        return results

    monkeypatch.setattr(database, "search_with_scores", fake_search)


async def run(message, **extra):
    state = {"message": message, "history": [], "user_date": "2026-01-20", **extra}
    return await agent.blossom_app.ainvoke(state)


async def test_paraphrased_question_is_answered_from_context(calls, monkeypatch):
    # Regression for the 'remember this device' hallucination
    mock_retrieval(monkeypatch, [(TRUSTED_DEVICE, 0.55)])
    out = await run("How often does 'remember this device' expire?")
    assert out["grounded"] is True
    assert "30 days" in calls[0]  # the chunk reached the prompt
    assert out["sources"] == [{"source": "Detailed_Security_Protocol.pdf", "page": 3}]
    assert "Detailed_Security_Protocol.pdf (Page 3)" in out["answer"]
    assert set(out["node_latency_ms"]) == {"route", "retrieve", "generate"}


async def test_off_topic_uses_fixed_fallback_and_never_calls_llm(calls, monkeypatch):
    mock_retrieval(monkeypatch, [(TRUSTED_DEVICE, 0.05)])
    out = await run("I want to open a new savings account")
    assert out["answer"] == agent.FALLBACK_MESSAGE
    assert out["grounded"] is False
    assert calls == []  # no LLM improvisation possible


async def test_no_documents_falls_back_even_if_on_topic(calls, monkeypatch):
    mock_retrieval(monkeypatch, [])
    out = await run("What are the password rules?")
    assert out["answer"] == agent.FALLBACK_MESSAGE
    assert calls == []


async def test_semantic_match_without_keyword_is_in_scope(calls, monkeypatch):
    mock_retrieval(monkeypatch, [(TRUSTED_DEVICE, 0.9)])
    out = await run("Why does the bank keep emailing me alerts when I travel?")
    assert out["grounded"] is True


async def test_greeting_skips_retrieval(calls, monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("retrieval must not run for greetings")

    monkeypatch.setattr(database, "search_with_scores", boom)
    out = await run("hello")
    assert out["topic"] == "Greeting"
    assert set(out["node_latency_ms"]) == {"route", "greet"}


async def test_holiday_reaches_the_prompt(calls, monkeypatch):
    mock_retrieval(monkeypatch, [(TRUSTED_DEVICE, 0.6)])
    await run("If I start a password reset today, when is the next step?", user_date="2026-01-19")
    assert "Martin Luther King, Jr. Day" in calls[0]


async def test_followup_inherits_previous_question(calls, monkeypatch):
    seen = []

    async def fake_search(query, k=4):
        seen.append(query)
        return [(TRUSTED_DEVICE, 0.6)]

    monkeypatch.setattr(database, "search_with_scores", fake_search)
    history = [{"role": "user", "content": "How do I recover my username?"},
               {"role": "assistant", "content": "..."}]
    await run("can you give me the steps?", history=history)
    assert seen == ["How do I recover my username?"]


async def test_retrieval_error_degrades_to_fallback(calls, monkeypatch):
    async def broken(query, k=4):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(database, "search_with_scores", broken)
    out = await run("What are the password rules?")
    assert out["answer"] == agent.FALLBACK_MESSAGE
