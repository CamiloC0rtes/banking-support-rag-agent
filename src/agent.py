"""Blossom support agent as an explicit LangGraph state machine.

Flow:

    route ──greeting──▶ greet ──────────────────────────▶ END
      │  ──farewell──▶ farewell ────────────────────────▶ END
      └──query──▶ retrieve ──grounded──▶ generate ───────▶ END
                       └────not grounded──▶ fallback ────▶ END

Design rules:
- Only `generate` writes policy content, and only from retrieved chunks.
- `fallback` is a fixed template: when we have no grounding we never ask the
  LLM to improvise an answer (this was the source of past hallucinations).
- Scope is decided by retrieval relevance OR a domain keyword match, so
  paraphrases ("remember this device", "codes") are not misrouted.
"""

import asyncio
import functools
import logging
import os
import re
import time
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import database

logger = logging.getLogger("blossom_agent")

# Global cache for MCP tools to reduce latency
_CACHED_HOLIDAYS: str | None = None
llm_client = ChatOpenAI(model=os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini"))

# Minimum relevance (0..1) for a chunk to count as grounding when no domain
# keyword matched. Calibrate with `python -m tests.eval.run_eval`.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
HISTORY_WINDOW = 10

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have verified information about that in Blossom's "
    "security documentation, so I'd rather not guess. Please contact Blossom "
    "customer support or check your banking portal for the next steps. "
    "I can help with login, passwords, MFA, trusted devices and account recovery."
)


class AgentState(TypedDict, total=False):
    # input
    message: str
    user_date: str | None
    history: list[dict]
    topic: str | None
    temperature: float | None
    top_p: float | None
    # working
    route: str
    search_query: str
    in_scope: bool
    docs: list
    scores: list[float]
    holiday_name: str | None
    # output
    answer: str
    sources: list[dict]
    contexts: list[str]
    grounded: bool
    node_latency_ms: dict


# -------------------------
# Observability
# -------------------------
def timed(func):
    """Record per-node latency in state['node_latency_ms'] and the logs."""

    @functools.wraps(func)
    async def wrapper(state: AgentState):
        start = time.perf_counter()
        result = await func(state)
        ms = round((time.perf_counter() - start) * 1000, 2)
        latencies = dict(state.get("node_latency_ms") or {})
        name = func.__name__.removesuffix("_node")
        latencies[name] = ms
        result["node_latency_ms"] = latencies
        logger.info(f"⏱️ node={name} latency_ms={ms}")
        return result

    return wrapper


# -------------------------
# MCP / Holiday Utilities
# -------------------------
async def call_mcp_holidays() -> str:
    """Fetches federal holiday data via MCP Server."""
    try:
        params = StdioServerParameters(command="python", args=["src/mcp_server.py"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                year = datetime.now().year
                result = await session.call_tool("get_federal_holidays", arguments={"year": year})
                return result.content[0].text
    except Exception as e:
        logger.error(f"MCP Connection Error: {e}")
        return "[]"


def parse_holiday(holidays_text: str, user_date_str: str) -> str | None:
    """Find the holiday name for a YYYY-MM-DD date in 'date: name' lines."""
    for line in (holidays_text or "").strip().split("\n"):
        if ":" in line:
            date_part, name_part = line.split(":", 1)
            if date_part.strip() == user_date_str:
                return name_part.strip()
    return None


async def fetch_holiday_name(user_date_str: str) -> str | None:
    """Returns the holiday name for the given date (cached MCP lookup)."""
    global _CACHED_HOLIDAYS
    if not _CACHED_HOLIDAYS:
        try:
            _CACHED_HOLIDAYS = await call_mcp_holidays()
        except Exception as e:
            logger.error(f"Error fetching holidays: {e}")
            return None
    try:
        return parse_holiday(_CACHED_HOLIDAYS, user_date_str)
    except Exception as e:
        logger.error(f"Error parsing holidays: {e}")
        return None


# -------------------------
# LLM Utilities
# -------------------------
async def stream_llm_response(system_prompt, history, user_message, temp=0.2, top_p=0.9):
    """Stream tokens from the chat model with a short history window."""
    messages = [SystemMessage(content=system_prompt)]
    for msg in history[-4:]:
        role = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(role(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))
    async for chunk in llm_client.astream(messages, temperature=temp, top_p=top_p):
        if chunk.content:
            yield chunk.content


async def complete(system_prompt, history, user_message, temp=0.2, top_p=0.9) -> str:
    chunks = [t async for t in stream_llm_response(system_prompt, history, user_message, temp, top_p)]
    return "".join(chunks).strip()


# -------------------------
# Routing helpers (pure, unit-tested)
# -------------------------
_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
_FAREWELLS = {"bye", "goodbye", "thanks", "thank you", "thx", "ty", "i'm done", "im done",
              "that's all", "no thanks", "stop"}
_STEPS_KEYWORDS = {"steps", "more", "instructions", "guide", "procedure", "how exactly"}

# Stems, so plurals and paraphrases match ("codes", "remember this device").
DOMAIN_PATTERNS = [
    r"log(ged|ging)? ?in", r"sign(ed|ing)? ?(in|up|on)", r"set ?up", r"passw", r"lock", r"unlock", r"access",
    r"mfa", r"2fa", r"two.factor", r"verif", r"\bcodes?\b", r"otp", r"token",
    r"remember", r"device", r"username", r"user ?name", r"secur", r"credential",
    r"yubikey", r"authent", r"identity", r"recover",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def is_small_talk(user_msg: str, phrases: set[str]) -> bool:
    """True only if the message is essentially just a greeting/farewell phrase.

    'hi, I forgot my username' must NOT be treated as a greeting.
    """
    norm = _normalize(user_msg)
    if not norm:
        return False
    for phrase in phrases:
        if norm == phrase or (norm.startswith(phrase) and len(norm.split()) <= len(phrase.split()) + 2):
            return True
    return False


def matches_domain(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in DOMAIN_PATTERNS)


def is_continuation(user_msg: str) -> bool:
    lower = user_msg.lower()
    return any(kw in lower for kw in _STEPS_KEYWORDS) and not matches_domain(user_msg)


def append_history(history: list, user_msg: str, answer: str) -> list:
    return (history + [{"role": "user", "content": user_msg},
                       {"role": "assistant", "content": answer}])[-HISTORY_WINDOW:]


def format_sources(docs: list) -> list[dict]:
    seen, sources = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources


def citation_footer(sources: list[dict]) -> str:
    if not sources:
        return ""
    refs = "; ".join(f"{s['source']} (Page {s['page']})" for s in sources)
    return f"\n\n—\nSources: {refs}"


def build_prompt(docs: list, day_name: str, user_date_str: str, holiday_name: str | None) -> str:
    """Grounded-answer system prompt: context only, explicit 'not in context' rule."""
    context_text = "\n".join(
        f"[Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')}]: {d.page_content}"
        for d in docs
    )
    holiday_message = (
        f"Today is a federal holiday ({holiday_name}). Manual reviews resume the next business day."
        if holiday_name
        else "Today is not a federal holiday."
    )
    return (
        "You are Blossom, a banking assistant specialized in login and security.\n"
        f"Today is {day_name}, {user_date_str}. {holiday_message}\n\n"
        "RULES:\n"
        "- Answer ONLY with facts stated in CONTEXT. Do not add procedures, links, buttons, "
        "time periods or numbers that are not in CONTEXT.\n"
        "- If CONTEXT does not answer the question (fully or partly), say clearly that the "
        "documentation doesn't cover that part and suggest contacting customer support.\n"
        "- Only help with login, password, MFA, trusted devices or account recovery.\n"
        "- If today is a holiday and the question involves a manual review, reset or recovery, "
        "mention that manual reviews resume the next business day and add "
        "'(source: federal holiday API)'.\n"
        "- Be warm and concise.\n\n"
        f"CONTEXT:\n{context_text}"
    )


# -------------------------
# Graph nodes
# -------------------------
@timed
async def route_node(state: AgentState):
    user_msg = state["message"].strip()
    history = state.get("history") or []

    if is_small_talk(user_msg, _FAREWELLS):
        return {"route": "farewell"}
    if is_small_talk(user_msg, _GREETINGS):
        return {"route": "greeting"}

    # "give me the steps" → inherit the previous question as search query
    search_query = user_msg
    if is_continuation(user_msg) and history:
        search_query = next((m["content"] for m in reversed(history) if m["role"] == "user"), user_msg)
        logger.info(f"Context inherited for follow-up: '{search_query}'")
    return {"route": "query", "search_query": search_query}


@timed
async def greet_node(state: AgentState):
    history = state.get("history") or []
    answer = await complete(
        "You are Blossom, a friendly banking assistant. The user greeted you. Reply briefly and "
        "mention you can help with login, passwords, MFA, trusted devices and account recovery. "
        "Do not give any policy details.",
        history, state["message"],
    )
    return {"answer": answer, "topic": "Greeting", "sources": [], "contexts": [], "grounded": False,
            "history": append_history(history, state["message"], answer)}


@timed
async def farewell_node(state: AgentState):
    answer = await complete(
        "You are Blossom, a friendly banking assistant. The user is ending the session. Write a short, "
        "warm goodbye inviting them back for login, password or security help.",
        state.get("history") or [], state["message"],
    )
    return {"answer": answer, "topic": None, "history": [], "sources": [], "contexts": [], "grounded": False}


@timed
async def retrieve_node(state: AgentState):
    query = state["search_query"]
    user_date = state.get("user_date") or datetime.now().strftime("%Y-%m-%d")

    async def _search():
        try:
            return await database.search_with_scores(query)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    results, holiday_name = await asyncio.gather(_search(), fetch_holiday_name(user_date))

    keyword_hit = matches_domain(query)
    top_score = max((s for _, s in results), default=0.0)
    in_scope = keyword_hit or top_score >= RELEVANCE_THRESHOLD
    logger.info(f"retrieval top_score={top_score:.3f} keyword_hit={keyword_hit} in_scope={in_scope}")

    return {
        "docs": [d for d, _ in results],
        "scores": [round(float(s), 4) for _, s in results],
        "in_scope": in_scope,
        "holiday_name": holiday_name,
    }


def grounding_gate(state: AgentState) -> str:
    return "generate" if state.get("in_scope") and state.get("docs") else "fallback"


@timed
async def generate_node(state: AgentState):
    history = state.get("history") or []
    user_msg = state["message"].strip()
    user_date = state.get("user_date") or datetime.now().strftime("%Y-%m-%d")
    day_name = datetime.strptime(user_date, "%Y-%m-%d").strftime("%A")
    docs = state["docs"]

    system_prompt = build_prompt(docs, day_name, user_date, state.get("holiday_name"))
    answer = await complete(system_prompt, history, user_msg,
                            temp=state.get("temperature") or 0.2, top_p=state.get("top_p") or 0.9)
    sources = format_sources(docs)
    return {
        "answer": f"{answer}{citation_footer(sources)}",
        "topic": docs[0].metadata.get("tags") or "Security/Login",
        "sources": sources,
        "contexts": [d.page_content for d in docs],
        "grounded": True,
        "history": append_history(history, user_msg, answer),
    }


@timed
async def fallback_node(state: AgentState):
    history = state.get("history") or []
    return {
        "answer": FALLBACK_MESSAGE,
        "topic": "Out of Scope",
        "sources": [],
        "contexts": [],
        "grounded": False,
        "history": append_history(history, state["message"].strip(), FALLBACK_MESSAGE),
    }


# -------------------------
# Graph definition
# -------------------------
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("greet", greet_node)
    builder.add_node("farewell", farewell_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("fallback", fallback_node)

    builder.set_entry_point("route")
    builder.add_conditional_edges(
        "route", lambda s: s["route"],
        {"greeting": "greet", "farewell": "farewell", "query": "retrieve"},
    )
    builder.add_conditional_edges("retrieve", grounding_gate, {"generate": "generate", "fallback": "fallback"})
    for node in ("greet", "farewell", "generate", "fallback"):
        builder.add_edge(node, END)
    return builder.compile()


blossom_app = build_graph()
