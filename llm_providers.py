"""Hosted LLM backends (Groq, Gemini) + helpers. Ollama stays in query.py.

Groq → Gemini: when ``RAG_LLM_BACKEND`` is groq and Groq returns 429/503, ``groq_*_with_gemini_fallback``
calls Gemini if a Gemini-compatible key is set: ``GEMINI_API_KEY`` (preferred), or ``GOOGLE_API_KEY`` /
``GENAI_API_KEY`` (disable with ``GROQ_GEMINI_FALLBACK=0``). Also falls back on HTTP 413 if Groq rejects payload size.

When both hosted calls fail, ``query.py`` can fall back to local Ollama (``RAG_HOSTED_FALLBACK_OLLAMA``).
Gemini 429: brief retries (``GEMINI_RETRY_ATTEMPTS``) then that Ollama path if enabled.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import httpx


def load_dotenv_if_present() -> None:
    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        # Minimal parser without python-dotenv
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def gemini_api_key() -> str:
    """First non-empty key among common Google AI Studio / Gemini env names."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return ""


def effective_llm_backend() -> str:
    """groq | gemini | ollama | openai — see RAG_LLM_BACKEND, auto if unset."""
    b = os.environ.get("RAG_LLM_BACKEND", "").strip().lower()
    if b in ("groq", "gemini", "ollama", "openai"):
        return b
    # Use local Ollama first when keys exist but you want zero API unless Ollama fails (no auto-up to API).
    if os.environ.get("RAG_LOCAL_FIRST", "").strip().lower() in ("1", "true", "yes", "ollama"):
        if os.environ.get("USE_OLLAMA", "1").strip().lower() in ("1", "true", "yes"):
            return "ollama"
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if gemini_api_key():
        return "gemini"
    if os.environ.get("USE_OLLAMA", "1").strip() in ("1", "true", "yes"):
        return "ollama"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    return "ollama"


def groq_model(fast: bool) -> str:
    if os.environ.get("GROQ_MODEL", "").strip():
        return os.environ["GROQ_MODEL"].strip()
    return "llama-3.1-8b-instant" if fast else "llama-3.3-70b-versatile"


def gemini_model(fast: bool) -> str:
    if os.environ.get("GEMINI_MODEL", "").strip():
        return os.environ["GEMINI_MODEL"].strip()
    return "gemini-2.0-flash" if fast else "gemini-2.0-flash"


def _groq_retry_attempts() -> int:
    if os.environ.get("GROQ_RETRY", "1").strip().lower() in ("0", "false", "no", "off"):
        return 1
    return max(1, int(os.environ.get("GROQ_RETRY_ATTEMPTS", "6")))


def _groq_backoff_sec(attempt_index: int) -> float:
    base = float(os.environ.get("GROQ_RETRY_BACKOFF_BASE", "1.6"))
    cap = float(os.environ.get("GROQ_RETRY_BACKOFF_MAX", "45"))
    return min(cap, base * (2**attempt_index) + random.uniform(0.0, 0.35))


def _groq_rate_limit_hint() -> str:
    return (
        "Groq የጥያቄ ወሰን (429) — ከአንድ ወደ ሁለት ደቂቃ በኋላ ይሞክሩ፣ ወይም "
        "`.env` ውስጥ `RAG_LLM_BACKEND=gemini` ወይም `ollama` ያዘጋጁ።"
    )


def openai_style_chat(
    messages: list[dict],
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.12 if "llama" in model.lower() else 0.15,
        "max_tokens": int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "2048")),
    }
    tmo = httpx.Timeout(connect=30.0, read=timeout_sec, write=120.0, pool=10.0)
    is_groq = "api.groq.com" in base_url
    attempts = _groq_retry_attempts() if is_groq else 1
    data: dict | None = None
    for attempt in range(attempts):
        with httpx.Client(timeout=tmo) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code in (429, 503) and attempt + 1 < attempts:
            time.sleep(_groq_backoff_sec(attempt))
            continue
        if r.status_code >= 400:
            if is_groq and r.status_code == 429 and attempt + 1 >= attempts:
                raise RuntimeError(_groq_rate_limit_hint()) from None
            if is_groq and r.status_code == 503 and attempt + 1 >= attempts:
                raise RuntimeError(
                    "Groq ሰርቨር ለአፊት ተወሰን (503) — ከጥቂት ጊዜ በኋላ ይሞክሩ ወይም RAG_LLM_BACKEND=gemini ይሞክሩ።"
                ) from None
            r.raise_for_status()
        data = r.json()
        break
    if data is None:
        raise RuntimeError("Groq: no response body after retries.")
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def groq_chat_messages(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    return openai_style_chat(
        messages,
        base_url="https://api.groq.com/openai/v1",
        api_key=key,
        model=groq_model(fast),
        timeout_sec=timeout_sec,
    )


def iter_groq_chat(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
):
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        yield "[groq] GROQ_API_KEY missing"
        return
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": groq_model(fast),
        "messages": messages,
        "temperature": 0.12,
        "max_tokens": int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "2048")),
        "stream": True,
    }
    tmo = httpx.Timeout(connect=30.0, read=timeout_sec, write=120.0, pool=10.0)
    attempts = _groq_retry_attempts()
    with httpx.Client(timeout=tmo) as client:
        for attempt in range(attempts):
            with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code in (429, 503) and attempt + 1 < attempts:
                    response.read()
                    time.sleep(_groq_backoff_sec(attempt))
                    continue
                if response.status_code >= 400:
                    response.read()
                    if response.status_code == 429:
                        raise RuntimeError(_groq_rate_limit_hint())
                    if response.status_code == 503 and attempt + 1 >= attempts:
                        raise RuntimeError(
                            "Groq ሰርቨር (503) — እንደገና ይሞክሩ ወይም ተለዋጭ LLM ይጠቀሙ።"
                        )
                    response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_s = line[6:].strip()
                    if data_s == "[DONE]":
                        break
                    try:
                        data = json.loads(data_s)
                    except json.JSONDecodeError:
                        continue
                    delta = (data.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece
                return


def _messages_to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    return "\n\n".join(system_parts).strip(), contents


def _gemini_retry_attempts() -> int:
    if os.environ.get("GEMINI_RETRY", "1").strip().lower() in ("0", "false", "no", "off"):
        return 1
    return max(1, min(6, int(os.environ.get("GEMINI_RETRY_ATTEMPTS", "3"))))


def _gemini_backoff_sec(attempt_index: int, response_text: str) -> float:
    """Parse retry hint from error JSON if present; else exponential backoff."""
    m = re.search(r"Please retry in ([0-9.]+)\s*s", response_text, re.I)
    if m:
        return min(60.0, float(m.group(1)) + random.uniform(0.15, 0.9))
    base = float(os.environ.get("GEMINI_RETRY_BACKOFF_BASE", "2.2"))
    cap = float(os.environ.get("GEMINI_RETRY_BACKOFF_MAX", "38"))
    return min(cap, base * (2**attempt_index) + random.uniform(0.0, 0.5))


def gemini_chat_messages(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> str:
    key = gemini_api_key()
    if not key:
        raise RuntimeError("No Gemini API key (set GEMINI_API_KEY or GOOGLE_API_KEY)")
    system_text, contents = _messages_to_gemini_contents(messages)
    if not contents:
        raise RuntimeError("No user/model messages for Gemini")
    model = gemini_model(fast)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", "2048")),
        },
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    tmo = httpx.Timeout(connect=30.0, read=timeout_sec, write=120.0, pool=10.0)
    attempts = _gemini_retry_attempts()
    last_detail = ""
    for attempt in range(attempts):
        with httpx.Client(timeout=tmo) as client:
            r = client.post(url, json=body)
        if r.status_code == 429 and attempt + 1 < attempts:
            last_detail = (r.text or "")[:2000]
            time.sleep(_gemini_backoff_sec(attempt, last_detail))
            continue
        if r.status_code >= 400:
            detail = (r.text or "")[:2000]
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {detail}")
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            raise RuntimeError(f"Gemini empty response: {data!r}"[:1500])
        parts = (cands[0].get("content") or {}).get("parts") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
    raise RuntimeError(f"Gemini: exhausted retries ({last_detail})")


def groq_gemini_fallback_enabled() -> bool:
    if os.environ.get("GROQ_GEMINI_FALLBACK", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return bool(gemini_api_key())


def should_fallback_groq_to_gemini(exc: BaseException) -> bool:
    if not groq_gemini_fallback_enabled():
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 503, 413)
    if isinstance(exc, RuntimeError):
        s = str(exc)
        if "429" in s or "503" in s or "413" in s:
            return True
        if "Too Many Requests" in s or "Payload Too Large" in s or "ወሰን" in s:
            return True
    return False


def groq_chat_messages_with_gemini_fallback(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> tuple[str, str]:
    """Returns (reply_text, backend_used): ``groq`` or ``gemini`` when Groq hits 429/503/413."""
    try:
        return (
            groq_chat_messages(messages, fast=fast, timeout_sec=timeout_sec),
            "groq",
        )
    except (RuntimeError, httpx.HTTPStatusError) as e:
        if should_fallback_groq_to_gemini(e):
            return (
                gemini_chat_messages(messages, fast=fast, timeout_sec=timeout_sec),
                "gemini",
            )
        raise


def iter_groq_chat_with_gemini_fallback(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
):
    """Stream Groq tokens, or one Gemini completion if Groq fails with 429/503/413."""
    try:
        yield from iter_groq_chat(messages, fast=fast, timeout_sec=timeout_sec)
    except (RuntimeError, httpx.HTTPStatusError) as e:
        if should_fallback_groq_to_gemini(e):
            yield gemini_chat_messages(messages, fast=fast, timeout_sec=timeout_sec)
            return
        raise


load_dotenv_if_present()
