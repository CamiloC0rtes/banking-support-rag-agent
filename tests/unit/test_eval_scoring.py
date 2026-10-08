"""The eval's scoring logic, checked offline against canned agent outputs."""

import re

import pytest

from tests.eval import run_eval
from tests.eval.run_eval import FALLBACK_MESSAGE, GAP_SIGNAL, Judgement, load_cases, run_case


class FakeApp:
    def __init__(self, out):
        self.out = out

    async def ainvoke(self, state):
        return self.out


class FakeJudge:
    def __init__(self, verdict):
        self.verdict = verdict

    async def ainvoke(self, prompt):
        return self.verdict


GOOD = Judgement(grounded=True)


def grounded_out(answer):
    return {"answer": answer + "\n\n—\nSources: x.pdf (Page 3)", "grounded": True,
            "sources": [{"source": "x.pdf", "page": 3}], "contexts": ["ctx"], "scores": [0.6]}


def test_golden_set_is_valid():
    cases = load_cases()
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["expect"] in {"answer", "gap", "fallback"}
        for p in c.get("must_include", []) + c.get("must_not", []):
            re.compile(p)


async def test_old_hallucination_is_caught(monkeypatch):
    # The exact answer the old agent produced, with no source
    old = {"answer": "The feature typically remains active until you clear your browser's cookies.",
           "grounded": False, "sources": [], "contexts": [], "scores": [0.4]}
    monkeypatch.setattr(run_eval, "blossom_app", FakeApp(old))
    case = {"id": "remember_device", "prompt": "q", "expect": "answer", "must_include": ["30 days"]}
    r = await run_case(case, FakeJudge(GOOD))
    assert not r["passed"]
    assert "answer without sources" in r["failures"]
    assert "missing fact /30 days/" in r["failures"]


async def test_correct_answer_passes(monkeypatch):
    monkeypatch.setattr(run_eval, "blossom_app", FakeApp(grounded_out("The cookie is valid for exactly 30 days.")))
    case = {"id": "remember_device", "prompt": "q", "expect": "answer", "must_include": ["30 days"]}
    assert (await run_case(case, FakeJudge(GOOD)))["passed"]


async def test_judge_flags_unsupported_claims(monkeypatch):
    monkeypatch.setattr(run_eval, "blossom_app", FakeApp(grounded_out("Click the Forgot Password link.")))
    case = {"id": "password_reset", "prompt": "q", "expect": "gap"}
    bad = Judgement(grounded=False, unsupported_claims=["There is a Forgot Password link"])
    r = await run_case(case, FakeJudge(bad))
    assert not r["passed"]
    assert "judge: unsupported claims" in r["failures"]


async def test_fallback_case(monkeypatch):
    out = {"answer": FALLBACK_MESSAGE, "grounded": False, "sources": [], "contexts": [], "scores": [0.1]}
    monkeypatch.setattr(run_eval, "blossom_app", FakeApp(out))
    r = await run_case({"id": "off", "prompt": "q", "expect": "fallback"}, FakeJudge(GOOD))
    assert r["passed"]


async def test_gap_admissions_from_judge_are_ignored(monkeypatch):
    monkeypatch.setattr(run_eval, "blossom_app", FakeApp(grounded_out("The documentation doesn't cover that part.")))
    noisy = Judgement(grounded=False, unsupported_claims=["The documentation doesn't cover that part."])
    r = await run_case({"id": "gap", "prompt": "q", "expect": "gap"}, FakeJudge(noisy))
    assert r["passed"]
    assert r["unsupported_claims"] == []


@pytest.mark.parametrize("text", [
    "The documentation doesn't cover password lockouts, so please contact customer support.",
    "Our security protocol does not specify a self-service unlock.",
])
def test_gap_signal(text):
    assert GAP_SIGNAL.search(text)
