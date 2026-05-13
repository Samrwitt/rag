"""Hosted LLM backends (Groq, Gemini) + helpers. Ollama stays in query.py."""

from __future__ import annotations

import json
import os
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


def effective_llm_backend() -> str:
    """groq | gemini | ollama | openai — see RAG_LLM_BACKEND, auto if unset."""
    b = os.environ.get("RAG_LLM_BACKEND", "").strip().lower()
    if b in ("groq", "gemini", "ollama", "openai"):
        return b
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if os.environ.get("GEMINI_API_KEY", "").strip():
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
    with httpx.Client(timeout=tmo) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
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
    with httpx.Client(timeout=tmo) as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
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


def gemini_chat_messages(
    messages: list[dict],
    *,
    fast: bool,
    timeout_sec: float,
) -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
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
    with httpx.Client(timeout=tmo) as client:
        r = client.post(url, json=body)
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


load_dotenv_if_present()
