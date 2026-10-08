"""Pure routing helpers: no network, no LLM."""

import pytest

from src.agent import (
    _FAREWELLS,
    _GREETINGS,
    citation_footer,
    format_sources,
    is_continuation,
    is_small_talk,
    matches_domain,
    parse_holiday,
)


@pytest.mark.parametrize(
    "msg",
    [
        # Regression: these were misrouted as out of scope and answered without context
        "How often does 'remember this device' expire?",
        "I changed phones and now my codes don't work. What should I do?",
        "I signed up, but I'm stuck — where do I finish my setup?",
        "Can I use a YubiKey?",
        "I forgot my username — how do I recover it?",
    ],
)
def test_domain_questions_are_in_scope(msg):
    assert matches_domain(msg)


@pytest.mark.parametrize("msg", ["I want to open a new savings account", "What's the mortgage rate?"])
def test_off_topic_has_no_domain_keyword(msg):
    assert not matches_domain(msg)


@pytest.mark.parametrize("msg", ["hi", "Hello!", "good morning", "hey there"])
def test_pure_greetings(msg):
    assert is_small_talk(msg, _GREETINGS)


def test_greeting_with_a_real_question_is_not_small_talk():
    assert not is_small_talk("hi, I forgot my username and can't log in", _GREETINGS)


@pytest.mark.parametrize("msg", ["thanks", "Thank you!", "bye", "I'm done"])
def test_farewells(msg):
    assert is_small_talk(msg, _FAREWELLS)


def test_continuation_only_without_new_topic():
    assert is_continuation("can you give me the steps?")
    assert not is_continuation("give me the steps to recover my username")


def test_parse_holiday():
    text = "2026-01-01: New Year's Day\n2026-01-19: Martin Luther King, Jr. Day"
    assert parse_holiday(text, "2026-01-19") == "Martin Luther King, Jr. Day"
    assert parse_holiday(text, "2026-01-20") is None
    assert parse_holiday("Holiday data currently unavailable.", "2026-01-19") is None


class _Doc:
    def __init__(self, source, page):
        self.metadata = {"source": source, "page": page}


def test_sources_are_deduplicated_and_all_cited():
    docs = [_Doc("a.pdf", 3), _Doc("a.pdf", 3), _Doc("a.pdf", 1)]
    sources = format_sources(docs)
    assert sources == [{"source": "a.pdf", "page": 3}, {"source": "a.pdf", "page": 1}]
    assert "a.pdf (Page 3); a.pdf (Page 1)" in citation_footer(sources)
    assert citation_footer([]) == ""
