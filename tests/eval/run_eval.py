"""Faithfulness evaluation for the Blossom agent.

Runs every case in golden.jsonl through the real graph (real retrieval + LLM)
and scores it on three axes:

1. Behavior: did it answer, admit a documentation gap, or use the safe fallback,
   as expected for that case?
2. Facts: required facts present (must_include) and known hallucinations absent (must_not).
3. Groundedness: an LLM judge checks every claim in the answer against the
   retrieved chunks and lists unsupported ones.

Usage:
    export OPENAI_API_KEY=...
    python -m tests.eval.run_eval            # writes eval_report.md / eval_report.json
    python -m tests.eval.run_eval --min-pass 0.9

Exit code is non-zero if the pass rate is below --min-pass.
Also prints the top retrieval score per case, to calibrate RELEVANCE_THRESHOLD.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
if __name__ == "__main__" and not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is required for the eval (set it in .env or the environment).")

from src.agent import FALLBACK_MESSAGE, blossom_app  # noqa: E402
from src.database import run_ingestion  # noqa: E402

HERE = Path(__file__).parent
GAP_SIGNAL = re.compile(
    r"(doesn't|does not|don't|do not) (cover|include|have|mention|specify|contain)|not (covered|specified|available)"
    r"|no (information|details)|contact(ing)? (blossom )?(customer )?support|reach(ing)? out to (customer )?support",
    re.I,
)


class Judgement(BaseModel):
    grounded: bool = Field(description="True if every factual claim is supported by CONTEXT")
    unsupported_claims: list[str] = Field(default_factory=list, description="Claims not supported by CONTEXT")


JUDGE_PROMPT = """You are a strict evaluator for a banking support RAG assistant.
Decide whether every factual claim in ANSWER is supported by CONTEXT or by VERIFIED SYSTEM FACTS.
Ignore greetings, empathy, advice to contact customer support, and the sources footer.
Statements that the documentation does not cover something are NOT unsupported claims.

VERIFIED SYSTEM FACTS:
{system_facts}

CONTEXT:
{context}

ANSWER:
{answer}"""


def load_cases():
    with open(HERE / "golden.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_footer(answer: str) -> str:
    return answer.split("\n\n—\nSources:")[0]


def system_facts(holiday) -> str:
    """Facts the agent legitimately gets from outside the PDF (MCP tool + business rules)."""
    if holiday:
        return (f"- Today is a US federal holiday: {holiday} (from the federal holiday API).\n"
                "- Bank rule: on holidays, manual reviews resume the next business day.")
    return "- Today is not a US federal holiday (from the federal holiday API)."


async def judge(llm, answer: str, contexts: list[str], holiday) -> Judgement:
    prompt = JUDGE_PROMPT.format(
        system_facts=system_facts(holiday),
        context="\n---\n".join(contexts) or "(empty)",
        answer=strip_footer(answer),
    )
    return await llm.ainvoke(prompt)


async def run_case(case: dict, judge_llm) -> dict:
    state = {"message": case["prompt"], "history": [], "user_date": case.get("user_date", "2026-01-20")}
    start = time.perf_counter()
    out = await blossom_app.ainvoke(state)
    latency = time.perf_counter() - start

    answer = out["answer"]
    body = strip_footer(answer)
    is_fallback = answer == FALLBACK_MESSAGE
    failures = []

    expect = case["expect"]
    if expect == "fallback" and not is_fallback:
        failures.append("expected safe fallback")
    if expect == "answer":
        if is_fallback:
            failures.append("fell back on an answerable question")
        elif not out.get("sources"):
            failures.append("answer without sources")
    if expect == "gap" and not (is_fallback or GAP_SIGNAL.search(body)):
        failures.append("did not acknowledge the documentation gap")

    for pattern in case.get("must_include", []):
        if not re.search(pattern, body, re.I):
            failures.append(f"missing fact /{pattern}/")
    for pattern in case.get("must_not", []):
        if re.search(pattern, body, re.I):
            failures.append(f"known hallucination /{pattern}/")

    unsupported = []
    if not is_fallback and out.get("grounded"):
        verdict = await judge(judge_llm, answer, out.get("contexts", []), out.get("holiday_name"))
        # Judges sometimes list "the docs don't cover X" as a claim; that's the desired behavior.
        unsupported = [c for c in verdict.unsupported_claims if not GAP_SIGNAL.search(c)]
        if unsupported:
            failures.append("judge: unsupported claims")

    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "expect": expect,
        "passed": not failures,
        "failures": failures,
        "unsupported_claims": unsupported,
        "top_score": max(out.get("scores") or [0.0]),
        "latency_s": round(latency, 3),
        "holiday": out.get("holiday_name"),
        "answer": answer,
    }


def write_report(results: list[dict], path_md: Path, path_json: Path):
    passed = sum(r["passed"] for r in results)
    latencies = sorted(r["latency_s"] for r in results)
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    summary = {
        "cases": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3),
        "p95_latency_s": p95,
        "median_latency_s": statistics.median(latencies),
    }
    path_json.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))

    lines = [
        "# Faithfulness eval",
        "",
        f"**{passed}/{len(results)} passed** · p95 latency {p95:.2f}s · median {summary['median_latency_s']:.2f}s",
        "",
        "| Case | Expect | Top score | Result | Notes |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        notes = "; ".join(r["failures"] + [f"unsupported: {c}" for c in r["unsupported_claims"]])
        lines.append(f"| {r['id']} | {r['expect']} | {r['top_score']:.2f} | {'✅' if r['passed'] else '❌'} | {notes} |")

    failed = [r for r in results if not r["passed"]]
    if failed:
        lines += ["", "## Failed answers", ""]
        for r in failed:
            answer = strip_footer(r["answer"]).replace("\n", " ")
            lines += [f"**{r['id']}** (holiday: {r['holiday'] or 'none'})", "", f"> {answer}", ""]
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass", type=float, default=0.9)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    run_ingestion(force_rebuild=False)
    judge_llm = ChatOpenAI(model=os.getenv("JUDGE_MODEL_NAME", "gpt-4o-mini"), temperature=0).with_structured_output(
        Judgement
    )

    results = []
    for case in load_cases():  # sequential: keeps latency numbers honest
        r = await run_case(case, judge_llm)
        results.append(r)
        print(f"{'✅' if r['passed'] else '❌'} {r['id']:<26} score={r['top_score']:.2f} {'; '.join(r['failures'])}")

    out = Path(args.out)
    summary = write_report(results, out / "eval_report.md", out / "eval_report.json")
    print(f"\n{summary['passed']}/{summary['cases']} passed · p95 {summary['p95_latency_s']:.2f}s")
    if summary["pass_rate"] < args.min_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
