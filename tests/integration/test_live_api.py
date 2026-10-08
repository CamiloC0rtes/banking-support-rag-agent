import json
import time

import httpx
import pytest

from src.agent import blossom_app, call_mcp_holidays
from src.database import get_active_retriever, run_ingestion

pytestmark = pytest.mark.integration

API_URL = "http://localhost:8000/chat"
SLA_THRESHOLD = 5.0
METRICS_FILE = "postdeploy_metrics.json"

PROMPTS = [
    "I got locked out after entering the wrong password. Can I unlock myself?",
    "What are the password rules? Can you list them quickly?",
    "Why do I keep getting verification codes when I log in?",
    "How often does 'remember this device' expire?",
    "I forgot my username — how do I recover it?",
    "I changed phones and now my codes don’t work. What should I do?",
    "Please help me reset my password safely.",
    "Can I unlock a phone‐banking user without calling support?",
    "I signed up, but I’m stuck — where do I finish my setup?",
    "If I start a password reset on a federal holiday, when should I expect the next step?"
]

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    run_ingestion(force_rebuild=False)

@pytest.mark.asyncio
async def test_chat_performance_and_holiday():
    latencies = []
    responses = []
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        for prompt in PROMPTS:
            start = time.perf_counter()
            r = await client.post(API_URL, json={"message": prompt, "user_date": "2026-01-19", "temperature": 0.2})
            duration = time.perf_counter() - start
            latencies.append(duration)
            assert r.status_code == 200
            answer = r.json().get("answer", "")
            responses.append({"prompt": prompt, "answer": answer, "latency_s": duration})
            print(f"✅ Prompt: {prompt[:30]}... | Latency: {duration:.2f}s")

        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[min(p95_index, len(latencies)-1)]
        print(f"\n🚀 FINAL P95 LATENCY: {p95_latency:.2f}s")
        assert p95_latency < SLA_THRESHOLD, f"SLA Violation: P95 is {p95_latency:.2f}s"

        # Verificar Holiday logic en el último prompt
        last_answer = responses[-1]["answer"].lower()
        assert "holiday" in last_answer or "delay" in last_answer

    # Guardar métricas en JSON
    metrics = {"latencies_s": latencies, "p95_s": p95_latency, "responses": responses}
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

@pytest.mark.asyncio
async def test_mcp_tool_call():
    try:
        result = await call_mcp_holidays()
        assert isinstance(result, str)
        assert "2026" in result
        print("\n✅ MCP Tool Test Passed")
    except Exception as e:
        pytest.fail(f"MCP Tool failed: {e}")

@pytest.mark.asyncio
async def test_retriever_observability():
    retriever = get_active_retriever()
    docs = await retriever.ainvoke("password reset steps")
    assert len(docs) > 0
    assert "source" in docs[0].metadata
    assert "page" in docs[0].metadata
    print("\n✅ Retriever Test Passed.")

@pytest.mark.asyncio
async def test_security_scope_guardrail():
    state = {
        "message": "I want to open a new savings account",
        "history": [],
        "user_date": "2026-01-11",
        "temperature": 0.0
    }
    response = await blossom_app.ainvoke(state)
    answer = response["answer"].lower()
    assert any(term in answer for term in ["support", "help center", "cannot help", "information"])
    print("\n✅ Security Guardrail Test Passed.")
# ------------------------------------------------------------------