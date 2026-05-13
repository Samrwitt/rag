"""Web snippets via DuckDuckGo (optional dependency: duckduckgo-search)."""

from __future__ import annotations

import os
from typing import Any

import httpx


def _ddgs_text(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise ImportError(
            "Install duckduckgo-search for better web grounding: pip install duckduckgo-search"
        ) from e

    out: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append(
                {
                    "title": r.get("title") or "",
                    "body": (r.get("body") or "").strip(),
                    "href": r.get("href") or "",
                }
            )
    return out


def fetch_web_snippets(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    """Return list of {title, body, href}. Empty on failure or empty query."""
    q = (query or "").strip()
    if not q:
        return []
    n = max(1, min(int(max_results), int(os.environ.get("RAG_WEB_MAX_RESULTS", "6"))))
    timeout = float(os.environ.get("RAG_WEB_HTTP_TIMEOUT", "25"))
    try:
        return _ddgs_text(q, n)
    except ImportError:
        pass
    except Exception:
        pass
    # Fallback: Wikipedia opensearch (no extra deps; narrow but stable)
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": q[:300],
            "limit": n,
            "namespace": 0,
            "format": "json",
        }
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        titles = (data[1] if len(data) > 1 else []) or []
        descs = (data[2] if len(data) > 2 else []) or []
        urls = (data[3] if len(data) > 3 else []) or []
        rows: list[dict[str, Any]] = []
        for i in range(min(len(titles), n)):
            rows.append(
                {
                    "title": titles[i],
                    "body": (descs[i] if i < len(descs) else "") or "",
                    "href": urls[i] if i < len(urls) else "",
                }
            )
        return rows
    except Exception:
        return []


def format_web_block(snippets: list[dict[str, Any]], start_index: int = 1) -> str:
    parts: list[str] = []
    for i, s in enumerate(snippets, start_index):
        tag = f"[W{i}]"
        title = (s.get("title") or "").strip()
        href = (s.get("href") or "").strip()
        body = (s.get("body") or "").strip()
        head = f"{tag} {title}"
        if href:
            head += f" — {href}"
        block = head
        if body:
            block += f"\n{body}"
        parts.append(block)
    return "\n\n".join(parts).strip()
