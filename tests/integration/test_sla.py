import time

import httpx
import pytest

pytestmark = pytest.mark.integration

API_URL = "http://localhost:8000/chat"
SLA_THRESHOLD = 5.0 

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

@pytest.mark.asyncio
async def test_performance_sla_p95():
    """Verify SLA compliance and Holiday Tool logic."""
    latencies = []
    
    async with httpx.AsyncClient() as client:
        # 1. Run the 10 prompts for Latency P95
        for p in PROMPTS:
            start_time = time.perf_counter()
            response = await client.post(
                API_URL,
                json={"message": p, "user_date": "2026-01-19", "temperature": 0.2},
                timeout=15.0
            )
            duration = time.perf_counter() - start_time
            
            assert response.status_code == 200
            latencies.append(duration)
            print(f"✅ Prompt: {p[:30]}... | Latency: {duration:.2f}s")

        # 2. Calculate P95 within the block
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[min(p95_index, len(latencies)-1)]
        print(f"\n🚀 FINAL P95 LATENCY: {p95_latency:.2f}s")

        # 3. Verify Holiday Logic (Re-checking the last response content)
        # We perform this inside the 'async with' to keep the client open
        holiday_check = await client.post(
            API_URL, 
            json={"message": PROMPTS[-1], "user_date": "2026-01-19"}
        )
        answer_text = holiday_check.json().get("answer", "").lower()
        
        # Verify grounding: Prompt #10 must detect the holiday (Jan 19, 2026 is MLK Day)
        assert "holiday" in answer_text or "delay" in answer_text
        
        # Final SLA Assertion
        assert p95_latency < SLA_THRESHOLD, f"SLA Violation: P95 is {p95_latency:.2f}s"