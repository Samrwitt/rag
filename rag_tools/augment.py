"""Merge KB retrieval with optional web + tool outputs (explicit prefixes + env)."""

from __future__ import annotations

import os
import re
from typing import Any

from rag_tools import weather as weather_mod
from rag_tools import web_search as web_mod


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _web_mode() -> str:
    return os.environ.get("RAG_WEB_MODE", "off").strip().lower()


def _auto_web_trigger(question: str) -> bool:
    if _env_flag("RAG_WEB_AUTO", "0"):
        return True
    q = question.strip()
    patterns = os.environ.get(
        "RAG_WEB_AUTO_REGEX",
        r"(https?://|የዛሬ|አዲስ ዜና|latest news|breaking|በድር ላይ)",
    )
    try:
        return bool(re.search(patterns, q, flags=re.IGNORECASE))
    except re.error:
        return False


def _kb_looks_sparse(hits: list[dict], *, min_hits: int = 2, max_dist: float | None = None) -> bool:
    if len(hits) < min_hits:
        return True
    if max_dist is None:
        raw = os.environ.get("RAG_WEB_SPARSE_MAX_DISTANCE", "").strip()
        max_dist = float(raw) if raw else None
    if max_dist is None:
        return False
    for h in hits[: min_hits]:
        d = h.get("distance")
        if d is None:
            continue
        try:
            if float(d) > max_dist:
                return True
        except (TypeError, ValueError):
            continue
    return False


def augment_kb_context(
    question: str,
    hits: list[dict],
    *,
    fast: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Returns (extra_user_text, tool_trace).

    Triggers:
    - ``!web <query>`` — always runs web search for ``query``.
    - ``!weather <location>`` — Open-Meteo current conditions.
    - Env ``RAG_WEB_MODE=always|auto|if_kb_sparse`` — augments without prefix (see env docs).
    """
    trace: list[dict[str, Any]] = []
    blocks: list[str] = []
    q = question.strip()

    # --- explicit commands (ASCII for easy mobile / Latin keyboard)
    if q.lower().startswith("!web "):
        sub = q[5:].strip()
        snippets = web_mod.fetch_web_snippets(
            sub, max_results=int(os.environ.get("RAG_WEB_MAX_RESULTS", "4"))
        )
        trace.append({"tool": "web_search", "args": {"query": sub}, "results": len(snippets)})
        if snippets:
            blocks.append(
                "የድር ፍለጋ ውጤት (ከመመሪያ ፋይሎች ውጭ፤ በ[W1] … ይጥቀሱ)፦\n"
                + web_mod.format_web_block(snippets, start_index=1)
            )
        else:
            blocks.append("የድር ፍለጋ ውጤት ባዶ ነው ወይም አልተሳካም።")
        return "\n\n".join(blocks), trace

    if q.lower().startswith("!weather "):
        loc = q[9:].strip()
        body = weather_mod.fetch_weather_summary(loc)
        trace.append({"tool": "weather_forecast", "args": {"location": loc}})
        blocks.append("የአየር ሁኔታ (Open-Meteo፣ ከውጭ)፦\n" + body)
        return "\n\n".join(blocks), trace

    if not _env_flag("RAG_TOOLS", "0"):
        return "", trace

    mode = _web_mode()
    want_web = False
    if mode == "always":
        want_web = True
    elif mode == "auto":
        want_web = _auto_web_trigger(q)
    elif mode == "if_kb_sparse":
        want_web = _kb_looks_sparse(hits)

    if want_web and _env_flag("RAG_WEB_ALLOW", "1"):
        snippets = web_mod.fetch_web_snippets(
            q, max_results=int(os.environ.get("RAG_WEB_MAX_RESULTS", "4"))
        )
        trace.append({"tool": "web_search", "args": {"query": q}, "results": len(snippets)})
        if snippets:
            blocks.append(
                "የድር ማጠናከሪያ (ከውጭ፤ ከመመሪያ ይለያል፤ [Wn] ይጥቀሱ)፦\n"
                + web_mod.format_web_block(snippets, start_index=1)
            )
        else:
            blocks.append("(ድር ማጠናከሪያ ባዶ ነው።)")

    if _env_flag("RAG_WEATHER_TOOL", "0"):
        # Very light heuristic: Amharic / English weather words + optional "በ <place>"
        m = re.search(
            r"(?:የአየር\s*ሁኔታ|weather|forecast)\s*(?:በ|at|in)\s*(.+)$",
            q,
            flags=re.IGNORECASE,
        )
        if m:
            loc = m.group(1).strip()[:120]
            if loc:
                body = weather_mod.fetch_weather_summary(loc)
                trace.append({"tool": "weather_forecast", "args": {"location": loc}})
                blocks.append(
                    "የአየር ሁኔታ (መሳሪያ፣ [W1] ከላይ ካለ በዚያ ቁጥር ይዝገቡ)፦\n" + body
                )

    return "\n\n".join(blocks).strip(), trace
