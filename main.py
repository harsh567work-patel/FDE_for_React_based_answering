"""
Full-Stack Documentation Engine — v2 (Optimized)
=================================================
Two-model architecture  +  deterministic KB scoring  +  source attribution.

Models
------
FAST  → qwen/qwen3.8-27b      guardrail · web eval · query refinement
POWER → openai/gpt-oss-120b   answer generation only
NONE  → KB scoring             deterministic — zero LLM calls
"""

import json
import os
import re
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

from knowledge_base.vector_store import init_vector_store

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━  CONFIG  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

FAST_MODEL   = os.getenv("FAST_MODEL",   "qwen/qwen3.8-27b")
POWER_MODEL  = os.getenv("POWER_MODEL",  "openai/gpt-oss-120b")

KB_SCORE_THRESHOLD  = 7        # ≥ 7 → answer from KB directly
WEB_SCORE_THRESHOLD = 6        # ≥ 6 → answer from KB + web
KB_TOP_K            = 5        # chunks retrieved from Qdrant

MAX_KB_CTX    = 3500           # chars sent to answer LLM
MAX_WEB_CTX   = 3500
MAX_DEEP_CTX  = 4500

ALLOWED_DOMAINS = (
    "React, FastAPI, Python, JavaScript, TypeScript, HTML, CSS, REST APIs, and Git"
)

# ━━━━━━━━━━━━━━━━━━━━━━━  STATE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineState(TypedDict):
    question: str
    is_relevant: bool
    kb_context: str            # formatted KB chunks (with source info)
    kb_score: int              # deterministic 1-10
    kb_score_detail: str       # human-readable breakdown
    web_context: str           # formatted web results
    web_score: int             # checklist-based 1-10
    web_score_detail: str
    deep_context: str          # deep-research results
    answer: str
    source: str                # pipeline path label


# ━━━━━━━━━━━━━━━━━━━━━━  LAZY SINGLETONS  ━━━━━━━━━━━━━━━━━━━

_fast_llm = None
_power_llm = None
_vs = None


def fast_llm() -> ChatGroq:
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = ChatGroq(model=FAST_MODEL, temperature=0)
    return _fast_llm


def power_llm() -> ChatGroq:
    global _power_llm
    if _power_llm is None:
        _power_llm = ChatGroq(model=POWER_MODEL, temperature=0)
    return _power_llm


def vs():
    global _vs
    if _vs is None:
        print("  ⏳ Loading embedding model + Qdrant connection …")
        _vs = init_vector_store()
    return _vs


# ━━━━━━━━━━━━━━━━━━━━━━━  HELPERS  ━━━━━━━━━━━━━━━━━━━━━━━━━━

def _trunc(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"


def _extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM output."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON block
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ━━━━━━━━━━━━━━━━━━━━━━━  NODE 1 — GUARDRAIL  ━━━━━━━━━━━━━━

_GUARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the intake gatekeeper for a specialized developer assistant.\n\n"
     "OUR STRICTLY SUPPORTED TECH STACK (AND NOTHING ELSE):\n"
     "- React (Components, Hooks, JSX, Virtual DOM, State, Props)\n"
     "- FastAPI (Endpoints, Pydantic, Dependencies, Async Python)\n"
     "- Core Full-Stack Web: Python, JavaScript, TypeScript, HTML tags (e.g. <div>, <span>), CSS, REST APIs, and Git.\n\n"
     "ANY OTHER LANGUAGE OR TOPIC (Java, C, C++, C#, Go, Rust, PHP, Ruby, Swift, non-tech) IS STRICTLY OUT OF BOUNDS.\n\n"
     "CLASSIFY INTO EXACTLY ONE WORD:\n"
     "- capability: Questions asking what technologies this assistant supports, or asking if it can answer a specific language (e.g. 'can it answer java', 'what do you support?', 'can you answer react?').\n"
     "- relevant: Questions strictly about React, FastAPI, Python, JavaScript, TypeScript, HTML tags (e.g. 'what is a div function', 'how to center a div'), CSS, REST APIs, or Git.\n"
     "- off_topic: ANY question about technologies outside our stack (e.g. Java, C, C++, Go, Rust) OR non-programming topics.\n\n"
     "Reply ONLY with one word: capability, relevant, or off_topic."),
    ("human", "{q}"),
])

_CAPABILITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are DevCopilot, a specialized developer assistant strictly limited to React and FastAPI.\n\n"
     "RULES:\n"
     "1. Our ONLY supported stack is: React, FastAPI, Python, JavaScript, TypeScript, HTML, CSS, REST APIs, and Git.\n"
     "2. We DO NOT support Java, C, C++, C#, Go, Rust, PHP, Ruby, Kotlin, Swift, or other external ecosystems.\n"
     "3. If the user asks if you support a technology OUTSIDE our stack (e.g. Java), clearly say: 'No, I do not answer questions about [tech]. I am strictly specialized for React and FastAPI (along with Python, JavaScript, TypeScript, HTML, CSS, REST APIs, and Git).'\n"
     "4. If the user asks if you support a technology INSIDE our stack (e.g. React, FastAPI, Python), clearly say: 'Yes, I support [tech]!' and give 2 quick examples.\n"
     "5. Keep the response under 90 words. Be direct, polite, and concise."),
    ("human", "{q}"),
])


def guardrail_node(state: PipelineState) -> dict:
    q = state["question"].strip()
    q_lower = q.lower()

    # Fast heuristic checks for immediate classification to minimize LLM token overhead
    tech_keywords = [
        "div", "span", "react", "fastapi", "python", "javascript", "typescript",
        "html", "css", "usememo", "usecallback", "useeffect", "usestate", "hook",
        "component", "pydantic", "endpoint", "rest api", "git"
    ]
    unsupported_keywords = [
        "java", "c++", "c#", "golang", "rust", "php", "ruby", "swift", "kotlin", "cobol"
    ]

    try:
        category = (_GUARD_PROMPT | fast_llm() | StrOutputParser()).invoke(
            {"q": q}
        ).strip().lower()
    except Exception:
        if any(term in q_lower for term in ["can you", "can it", "do you support", "what stack", "what languages", "what can you"]):
            category = "capability"
        elif any(u in q_lower for u in unsupported_keywords):
            category = "off_topic"
        elif any(re.search(r'\b' + re.escape(w) + r'\b', q_lower) for w in tech_keywords):
            category = "relevant"
        else:
            category = "off_topic"

    if "capability" in category:
        try:
            ans = (_CAPABILITY_PROMPT | fast_llm() | StrOutputParser()).invoke({"q": q})
        except Exception:
            ans = (
                "I am strictly specialized for **React** and **FastAPI** full-stack development "
                "(including Python, JavaScript, TypeScript, HTML/CSS, REST APIs, and Git). "
                "I do not answer questions about external ecosystems like Java, C++, or Go."
            )
        return {
            "is_relevant": False,
            "answer": ans,
            "source": "capability",
        }

    if "relevant" in category:
        return {"is_relevant": True}

    return {
        "is_relevant": False,
        "answer": (
            "⛔ This question is outside my supported tech stack. I do not answer questions about external or non-supported topics. "
            "My scope is strictly limited to **React** and **FastAPI** full-stack documentation "
            "(including Python, JavaScript, TypeScript, HTML/CSS, REST APIs, and Git)."
        ),
        "source": "guardrail",
    }


# ━━━━━━━━━━━━━━━━━━━━━  NODE 2 — KB SEARCH  ━━━━━━━━━━━━━━━━

def kb_search_node(state: PipelineState) -> dict:
    results = vs().similarity_search_with_score(state["question"], k=KB_TOP_K)

    if not results:
        return {"kb_context": "", "kb_score": 0, "kb_score_detail": "No chunks found"}

    q_tokens = set(state["question"].lower().split())
    scored = []

    for doc, sim in results:
        # keyword overlap
        d_tokens = set(doc.page_content.lower().split())
        kw = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        # content density (penalise very short chunks)
        density = min(len(doc.page_content) / 500, 1.0)
        # composite
        comp = sim * 0.65 + kw * 0.20 + density * 0.15
        scored.append({
            "doc": doc,
            "sim": round(sim, 4),
            "kw": round(kw, 4),
            "density": round(density, 2),
            "composite": round(comp, 4),
        })

    scored.sort(key=lambda x: x["composite"], reverse=True)

    # final score from top-3 composites → 1-10
    top3 = scored[:3]
    avg = sum(s["composite"] for s in top3) / len(top3)
    final = max(1, min(10, round(avg * 12)))

    # detail string
    detail_lines = [f"avg_composite={avg:.3f} → score={final}/10"]
    for i, s in enumerate(top3, 1):
        detail_lines.append(
            f"  #{i}  sim={s['sim']}  kw={s['kw']}  den={s['density']}  "
            f"comp={s['composite']}  [{s['doc'].metadata.get('technology','')}]"
        )
    detail = "\n".join(detail_lines)

    # build context with source attribution
    parts = []
    for i, s in enumerate(scored[:KB_TOP_K], 1):
        d = s["doc"]
        tech = d.metadata.get("technology", "?")
        path = d.metadata.get("path", "")
        url  = d.metadata.get("url", "")
        parts.append(
            f"[KB Source {i}] {tech} — {path}  (relevance: {s['composite']})\n"
            f"URL: {url}\n"
            f"{d.page_content}"
        )

    return {
        "kb_context": _trunc("\n\n---\n\n".join(parts), MAX_KB_CTX),
        "kb_score": final,
        "kb_score_detail": detail,
    }


# ━━━━━━━━━━━━━━━━━━━  NODE 3 — KB DIRECT ANSWER  ━━━━━━━━━━━

_KB_ANS = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise enterprise documentation assistant specialized strictly in React and FastAPI.\n"
     "• Focus strictly on React, FastAPI, and web development. Do NOT discuss unrelated languages (e.g. C, C++, R, Java).\n"
     "• Be direct, concise, and under 250 words to conserve tokens.\n"
     "• When asked about 'div' or 'div function', clarify that <div> is an HTML/React layout container element, not a mathematical function.\n"
     "• COMPARISON / VS / DIFFERENTIATE QUERIES: Whenever comparing technologies, functions, or concepts, "
     "structure the comparison as a clean, complete Markdown table with clear column headers "
     "(e.g., | Feature | Item A | Item B |) and 3-5 rows. Ensure every row starts and ends with a '|'.\n"
     "• At the end list every KB Source you used (number + URL).\n"
     "• Do NOT fabricate information."),
    ("human", "Q: {q}\n\nContext:\n{ctx}"),
])


def kb_answer_node(state: PipelineState) -> dict:
    answer = (_KB_ANS | power_llm() | StrOutputParser()).invoke({
        "q": state["question"],
        "ctx": state["kb_context"],
    })
    return {"answer": answer, "source": "knowledge_base"}


# ━━━━━━━━━━━━━━━━━━  NODE 4 — TAVILY WEB SEARCH  ━━━━━━━━━━━

def _fmt_tavily(raw) -> str:
    if isinstance(raw, list):
        parts = []
        for r in raw:
            if isinstance(r, dict):
                parts.append(f"[Web] {r.get('url','')}\n{r.get('content','')}")
            else:
                parts.append(str(r))
        return "\n\n".join(parts)
    return str(raw)


def tavily_search_node(state: PipelineState) -> dict:
    try:
        raw = TavilySearch(max_results=5).invoke(state["question"])
        txt = _fmt_tavily(raw)
    except Exception as e:
        txt = f"Search error: {e}"
    return {"web_context": _trunc(txt, MAX_WEB_CTX)}


# ━━━━━━━━━━━━━━━  NODE 5 — WEB EVAL (CHECKLIST)  ━━━━━━━━━━━

_WEB_EVAL = ChatPromptTemplate.from_messages([
    ("system",
     "Evaluate if these search results can answer the question.\n"
     "Reply with ONLY valid JSON — no other text:\n"
     '{{"addresses_question": true/false, "has_specifics": true/false, '
     '"authoritative": true/false, "sufficient": true/false}}'),
    ("human", "Question: {q}\n\nResults:\n{ctx}"),
])


def web_evaluate_node(state: PipelineState) -> dict:
    raw = (_WEB_EVAL | fast_llm() | StrOutputParser()).invoke({
        "q": state["question"],
        "ctx": state.get("web_context", ""),
    })
    criteria = _extract_json(raw)
    true_count = sum(1 for v in criteria.values() if v is True)
    score = max(1, round(true_count * 2.5))

    detail = " | ".join(f"{k}={'✓' if v else '✗'}" for k, v in criteria.items())
    detail += f" → {true_count}/4 → score={score}/10"

    return {"web_score": score, "web_score_detail": detail}


# ━━━━━━━━━━━━━━━  NODE 6 — WEB-AUGMENTED ANSWER  ━━━━━━━━━━━

_WEB_ANS = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise enterprise documentation assistant specialized strictly in React and FastAPI.\n"
     "• Focus strictly on React, FastAPI, and web development. Do NOT discuss unrelated languages (e.g. C, C++, R, Java).\n"
     "• Be direct, concise, and under 250 words to conserve tokens.\n"
     "• When asked about 'div' or 'div function', clarify that <div> is an HTML/React layout container element, not a mathematical function.\n"
     "• COMPARISON / VS / DIFFERENTIATE QUERIES: Whenever comparing technologies, functions, or concepts, "
     "structure the comparison as a clean, complete Markdown table with clear column headers "
     "(e.g., | Feature | Item A | Item B |) and 3-5 rows. Ensure every row starts and ends with a '|'.\n"
     "• At the end list ALL sources used (KB Source # + URL, Web URLs).\n"
     "• Do NOT fabricate information."),
    ("human", "Q: {q}\n\nKB Context:\n{kb}\n\nWeb Context:\n{web}"),
])


def web_answer_node(state: PipelineState) -> dict:
    answer = (_WEB_ANS | power_llm() | StrOutputParser()).invoke({
        "q": state["question"],
        "kb": state.get("kb_context", "— none —"),
        "web": state.get("web_context", "— none —"),
    })
    return {"answer": answer, "source": "kb_and_web"}


# ━━━━━━━━━━━━━━━━  NODE 7 — DEEP RESEARCH  ━━━━━━━━━━━━━━━━━

_REFINE = ChatPromptTemplate.from_messages([
    ("system", "Rewrite into a precise technical search query focusing strictly on web/React/FastAPI. Reply with ONLY the query."),
    ("human", "{q}"),
])


def deep_research_node(state: PipelineState) -> dict:
    refined = (_REFINE | fast_llm() | StrOutputParser()).invoke(
        {"q": state["question"]}
    ).strip()

    try:
        raw = TavilySearch(max_results=5).invoke(refined)
        txt = _fmt_tavily(raw)
    except Exception as e:
        txt = f"Deep search error: {e}"

    return {"deep_context": _trunc(txt, MAX_DEEP_CTX)}


# ━━━━━━━━━━━━━━━━━━  NODE 8 — FINAL ANSWER  ━━━━━━━━━━━━━━━━

_FINAL_ANS = ChatPromptTemplate.from_messages([
    ("system",
     "You are a concise enterprise documentation assistant specialized strictly in React and FastAPI.\n"
     "• Focus strictly on React, FastAPI, and web development. Do NOT discuss unrelated languages (e.g. C, C++, R, Java).\n"
     "• Be direct, concise, and under 250 words to conserve tokens.\n"
     "• When asked about 'div' or 'div function', clarify that <div> is an HTML/React layout container element, not a mathematical function.\n"
     "• COMPARISON / VS / DIFFERENTIATE QUERIES: Whenever comparing technologies, functions, or concepts, "
     "structure the comparison as a clean, complete Markdown table with clear column headers "
     "(e.g., | Feature | Item A | Item B |) and 3-5 rows. Ensure every row starts and ends with a '|'.\n"
     "• At the end list ALL sources (KB Source # + URL, Web URLs).\n"
     "• Do NOT fabricate information."),
    ("human", "Q: {q}\n\nAll Context:\n{ctx}"),
])


def final_answer_node(state: PipelineState) -> dict:
    sections = []
    if state.get("kb_context"):
        sections.append(state["kb_context"])
    if state.get("web_context"):
        sections.append(state["web_context"])
    if state.get("deep_context"):
        sections.append(state["deep_context"])

    ctx = "\n\n---\n\n".join(sections) or "No context found."

    answer = (_FINAL_ANS | power_llm() | StrOutputParser()).invoke({
        "q": state["question"],
        "ctx": _trunc(ctx, MAX_DEEP_CTX),
    })
    return {"answer": answer, "source": "deep_research"}


# ━━━━━━━━━━━━━━━━━━━━━  ROUTING  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_guardrail(state: PipelineState) -> str:
    return "kb_search" if state.get("is_relevant") else END


def route_kb(state: PipelineState) -> str:
    return "kb_answer" if state.get("kb_score", 0) >= KB_SCORE_THRESHOLD else "tavily_search"


def route_web(state: PipelineState) -> str:
    return "web_answer" if state.get("web_score", 0) >= WEB_SCORE_THRESHOLD else "deep_research"


# ━━━━━━━━━━━━━━━━━━━  GRAPH ASSEMBLY  ━━━━━━━━━━━━━━━━━━━━━━

def build_pipeline():
    g = StateGraph(PipelineState)

    g.add_node("guardrail",     guardrail_node)
    g.add_node("kb_search",     kb_search_node)
    g.add_node("kb_answer",     kb_answer_node)
    g.add_node("tavily_search", tavily_search_node)
    g.add_node("web_evaluate",  web_evaluate_node)
    g.add_node("web_answer",    web_answer_node)
    g.add_node("deep_research", deep_research_node)
    g.add_node("final_answer",  final_answer_node)

    g.set_entry_point("guardrail")

    g.add_conditional_edges("guardrail", route_guardrail,
                            {"kb_search": "kb_search", END: END})

    g.add_conditional_edges("kb_search", route_kb,
                            {"kb_answer": "kb_answer", "tavily_search": "tavily_search"})

    g.add_edge("kb_answer", END)

    g.add_edge("tavily_search", "web_evaluate")

    g.add_conditional_edges("web_evaluate", route_web,
                            {"web_answer": "web_answer", "deep_research": "deep_research"})

    g.add_edge("web_answer", END)
    g.add_edge("deep_research", "final_answer")
    g.add_edge("final_answer", END)

    return g.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━  CLI  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_LABELS = {
    "guardrail":      "⛔ Guardrail",
    "knowledge_base": "📚 Knowledge Base",
    "kb_and_web":     "📚 KB + 🌐 Web",
    "deep_research":  "🔬 Deep Research",
}


def main():
    print("=" * 60)
    print("  Full-Stack Documentation Engine  v2")
    print(f"  Fast: {FAST_MODEL}  |  Power: {POWER_MODEL}")
    print("=" * 60)
    print("Type 'quit' to stop.\n")

    pipeline = build_pipeline()

    while True:
        try:
            q = input("\n📝 Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        t0 = time.time()
        result = pipeline.invoke({"question": q})
        elapsed = time.time() - t0

        src = result.get("source", "?")
        print(f"\n  Route     : {_LABELS.get(src, src)}")
        print(f"  Time      : {elapsed:.1f}s")

        if result.get("kb_score"):
            print(f"  KB Score  : {result['kb_score']}/10  "
                  f"{'✅ Strong' if result['kb_score'] >= KB_SCORE_THRESHOLD else '→ Fallback'}")
        if result.get("kb_score_detail"):
            for line in result["kb_score_detail"].split("\n"):
                print(f"    {line}")

        if result.get("web_score"):
            print(f"  Web Score : {result['web_score']}/10  "
                  f"{'✅ Good' if result['web_score'] >= WEB_SCORE_THRESHOLD else '→ Deep Research'}")
        if result.get("web_score_detail"):
            print(f"    {result['web_score_detail']}")

        print("─" * 60)
        print(result.get("answer", "No answer generated."))
        print("─" * 60)


if __name__ == "__main__":
    main()
