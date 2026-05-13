"""Query Amharic RAG: retrieve from Chroma + answer via Ollama or OpenAI-compatible API.

Retrieval: dense embeddings (Chroma) + optional BM25 sidecar with RRF fusion (see hybrid_retrieval).
Optional web/tools: see rag_tools/augment.py (RAG_TOOLS, RAG_WEB_MODE, !web / !weather prefixes).
Optional dynamic_layer: run ``python dynamic_layer.py`` then ``python ingest.py`` merges ``data/chunks/dynamic_context_chunks.jsonl`` when present (``--no-dynamic`` to skip).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import chromadb
import httpx

from embeddings import encode_query
from qa_context import qa_context_text
from rag_tools import augment_kb_context

# Short system line in fast mode saves prompt tokens / latency
SYSTEM_AM = (
    "አንተ የኢትዮጵያ ግብርና እና የተፈጥሮ ሀብት ባለሙያ አማካሪ ነህ። "
    "የተሰጠህን መረጃ ብቻ በመጠቀም በአማርኛ ግልጽና ትክክለኛ መልስ ስጥ። "
    "ከመረጃው ውጭ ከሆነ በግልጽ 'በዚህ መረጃ ውስጥ መልስ አልተገኘም' በማለት ግልጽ አድርግ። "
    "የእያንዳንዱን ክፍል ምንጭ በመግለጽ በቁጥር ጥቅስ (ለምሳሌ፦ «…» [1])።"
)
SYSTEM_AM_FAST = (
    "በአማርኛ ትክክለኛ መልስ ስጥ። መረጃውን ብቻ ተጠቀም። ከውጭ ከሆነ በግልጽ ብቻ ንገር። "
    "አስፈላጊ ከሆነ ከመረጃው ጋር በሚስማማ ቁጥር [1] [2] ጥቅስ።"
)
SYSTEM_AUX_WEB = (
    " ከ[1] የሚጀመሩ ክፍሎች ከየእኛ መመሪያ ቤት ናቸው። [W1] … ከድር ወይም ከመሳሪያ ሲሆኑ "
    "አስቀድመህ የመመሪያውን መልስ አረጋግጥ፤ የድርን መረጃ በጥንቃቄ ተጠቀም።"
)


def system_for_rag(
    fast: bool, *, has_aux_context: bool, follow_up: bool = False
) -> str:
    base = SYSTEM_AM_FAST if fast else SYSTEM_AM
    if has_aux_context:
        base = base + SYSTEM_AUX_WEB
    if follow_up:
        base += (
            " የቅርብ ጥያቄዎችንና ከላይ ያለውን መረጃ ብቻ በመጠቀም መልስ። "
            "«ከዚህ» ወይም «በላይ» ካለ ቀዳሚውን ጥያቄና መልስ አጣምር። ከመረጃ ውጭ አትወ።"
        )
    return base


def read_embed_model(db: Path) -> str | None:
    p = db / "embed_model.txt"
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8").strip() or None


def _fast_mode(cli_quality: bool) -> bool:
    if cli_quality:
        return False
    return os.environ.get("RAG_MODE", "fast").strip().lower() not in (
        "quality",
        "slow",
        "accurate",
    )


def default_top_k(fast: bool) -> int:
    return 4 if fast else 6


def default_chat_model(fast: bool) -> str:
    if os.environ.get("OLLAMA_MODEL", "").strip():
        return os.environ["OLLAMA_MODEL"].strip()
    # ~3s target: small Amharic-tuned 1B. Quality: larger multilingual model.
    return "amharic-llama-1b-safe:latest" if fast else "qwen3:4b-instruct"


@contextmanager
def rag_runtime_env(**updates: str | None):
    """Temporarily set process env for RAG tools / web (e.g. Streamlit toggles)."""
    saved: dict[str, str | None] = {}
    try:
        for k, v in updates.items():
            saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def ollama_options(fast: bool) -> dict:
    raw = os.environ.get("OLLAMA_OPTIONS_JSON", "").strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("Warning: OLLAMA_OPTIONS_JSON invalid JSON, using preset.", file=sys.stderr)
    if fast:
        return {
            "temperature": 0.08,
            "top_p": 0.85,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "3072")),
            "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "280")),
        }
    # Smaller default ctx avoids Ollama 500 (VRAM/RAM) on laptops with iGPU + 4B models
    return {
        "temperature": 0.05,
        "top_p": 0.9,
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
        "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "512")),
    }


def retrieve(
    collection,
    query: str,
    top_k: int,
    embed_model: str | None,
    db: Path | None = None,
) -> list[dict]:
    qv = encode_query(query, model=embed_model)[0].tolist()
    hybrid_on = os.environ.get("RAG_HYBRID", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if db is not None and hybrid_on:
        try:
            from hybrid_retrieval import hybrid_retrieve, sidecar_exists

            if sidecar_exists(db):
                return hybrid_retrieve(collection, db, query, qv, top_k)
        except Exception as e:
            print(f"Warning: hybrid retrieval failed, using dense only: {e}", file=sys.stderr)

    res = collection.query(
        query_embeddings=[qv],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[dict] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        text = qa_context_text(meta or {}, doc)
        rows.append({"text": text, "meta": meta or {}, "distance": dist})
    return rows


def build_context(chunks: list[dict], max_chars: int) -> str:
    parts: list[str] = []
    n = 0
    for i, c in enumerate(chunks, 1):
        meta = c["meta"]
        src = meta.get("source", "?")
        kind = meta.get("kind", "")
        page = meta.get("page", "")
        head = f"[{i}] ምንጭ: {src}"
        if kind == "pdf" and page:
            head += f" ገጽ {page}"
        head += f" — ለመልስ ውስጥ ጥቅስ: [{i}]"
        block = f"{head}\n{c['text']}\n"
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n".join(parts).strip()


def format_source_rows(hits: list[dict]) -> list[dict]:
    return [
        {
            "source": h["meta"].get("source"),
            "kind": h["meta"].get("kind"),
            "page": h["meta"].get("page"),
            "distance": h["distance"],
            "preview": h["text"][:240] + ("…" if len(h["text"]) > 240 else ""),
        }
        for h in hits
    ]


def retrieval_query_for(question: str, conversation: list[dict] | None) -> str:
    """Build the string used for dense/hybrid *embedding* search.

    Follow-ups like «ከዚህ በላይ ምን …?» have no topic words alone; we fold in recent
    user/assistant text so Chroma/BM25 see the same subject as the chat.
    """
    q = (question or "").strip()
    if not conversation:
        return q
    if os.environ.get("RAG_RETRIEVAL_USE_CONVERSATION", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return q
    parts: list[str] = []
    for m in conversation[-6:]:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = (m.get("content") or "").strip()
        if not c:
            continue
        parts.append(f"{role}: {c[:900]}")
    if not parts:
        return q
    hist = "\n".join(parts)
    cap = int(os.environ.get("RAG_RETRIEVAL_CONTEXT_CHARS", "3200"))
    blob = f"ውይይት ታሪክ:\n{hist}\n\nአሁን የተጠየቀው:\n{q}"
    while len(blob) > cap and hist:
        hist = hist[max(200, len(hist) // 5) :]
        blob = f"ውይይት ታሪክ:\n{hist}\n\nአሁን የተጠየቀው:\n{q}"
    return blob.strip()


def build_rag_pack(
    question: str,
    db: Path,
    top_k: int,
    *,
    fast: bool,
    conversation: list[dict] | None = None,
) -> dict:
    """Retrieve + optional web/tools + user prompt block (sync + streaming)."""
    client = chromadb.PersistentClient(path=str(db))
    collection = client.get_collection("amharic_rag")
    embed_model = read_embed_model(db)
    env_m = os.environ.get("RAG_EMBED_MODEL", "").strip()
    if embed_model and env_m and env_m != embed_model:
        print(
            f"Warning: RAG_EMBED_MODEL={env_m!r} != indexed {embed_model!r}; using indexed model.",
            file=sys.stderr,
        )
    rq = retrieval_query_for(question, conversation)
    hits = retrieve(collection, rq, top_k=top_k, embed_model=embed_model, db=db)
    extra_ctx, tool_trace = augment_kb_context(question, hits, fast=fast)
    base_max = int(os.environ.get("RAG_CONTEXT_CHARS", "4200" if fast else "9000"))
    reserved = min(len(extra_ctx) + 400, 3600) if extra_ctx.strip() else 0
    max_chars = max(900, base_max - reserved)
    ctx = build_context(hits, max_chars=max_chars)
    has_aux = bool(extra_ctx.strip())
    aux_block = f"ተጨማሪ (ድር / መሳሪያ — ከመመሪያ ቤት ይለያል፤ [W1] …)፦\n{extra_ctx}\n\n" if has_aux else ""
    user_block = (
        f"ጥያቄ፦ {question.strip()}\n\n"
        f"መረጃ ከመመሪያ ቤት (ቁጥር [1] [2] …)፦\n{ctx}\n\n"
        f"{aux_block}"
        "በአማርኛ አጫጭን መልስ ስጥ፤ ከመረጃ የወጡ ክፍሎችን በ [1] ወይም [2]፣ ከተጨማሪው በ [W1] … አድርግ።"
        if fast
        else (
            f"ጥያቄ፦ {question.strip()}\n\n"
            f"የማመሳከሪያ መረጃ ከመመሪያ ቤት፦\n{ctx}\n\n"
            f"{aux_block}"
            "ከላይ ባለው መረጃ መሰረት ጥያቄውን በአማርኛ መልስ። "
            "የእያንዳንዱን ክፍል ምንጭ በቁጥር [1] [2] ወይም [W1] ጥቅስ።"
        )
    )
    retrieval = (
        "hybrid"
        if hits and hits[0].get("rrf_rank") is not None
        else "dense"
    )
    return {
        "hits": hits,
        "user_block": user_block,
        "retrieval": retrieval,
        "tool_trace": tool_trace,
        "has_aux_context": has_aux,
        "retrieval_query": rq,
    }


def _ollama_messages(
    system: str,
    conversation: list[dict] | None,
    user_block: str,
) -> list[dict]:
    """Prior user/assistant turns, then current RAG user message."""
    messages: list[dict] = [{"role": "system", "content": system}]
    if conversation:
        for m in conversation[-12:]:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            c = (m.get("content") or "").strip()
            if not c:
                continue
            messages.append({"role": role, "content": c[:12000]})
    messages.append({"role": "user", "content": user_block})
    return messages


def ollama_chat_messages(
    messages: list[dict],
    model: str,
    base_url: str,
    *,
    options: dict,
    timeout_sec: float,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    tmo = httpx.Timeout(connect=20.0, read=timeout_sec, write=120.0, pool=10.0)
    try:
        with httpx.Client(timeout=tmo) as client:
            r = client.post(url, json=payload)
    except httpx.ReadTimeout as e:
        raise RuntimeError(
            f"Ollama read timed out after {timeout_sec:g}s (model may be loading on CPU/GPU). "
            f"Try: export OLLAMA_HTTP_TIMEOUT=180  or  ollama run {model}  once to warm the model."
        ) from e
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url!r}. Is `ollama serve` running?"
        ) from e
    if r.status_code >= 400:
        detail = (r.text or "").strip()[:2500]
        raise RuntimeError(
            f"Ollama HTTP {r.status_code} for model {model!r}. "
            f"Often: out of memory — try export OLLAMA_NUM_CTX=2048 or a smaller model. Body:\n{detail}"
        )
    data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()


def iter_ollama_chat(
    messages: list[dict],
    model: str,
    base_url: str,
    *,
    options: dict,
    timeout_sec: float,
):
    """Yield text fragments from Ollama streaming /api/chat."""
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    tmo = httpx.Timeout(connect=20.0, read=timeout_sec, write=120.0, pool=10.0)
    with httpx.Client(timeout=tmo) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                piece = (data.get("message") or {}).get("content") or ""
                if piece:
                    yield piece


def ollama_chat(
    model: str,
    user_prompt: str,
    base_url: str,
    *,
    options: dict,
    system: str,
    timeout_sec: float,
) -> str:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    return ollama_chat_messages(
        msgs, model, base_url, options=options, timeout_sec=timeout_sec
    )


def openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_AM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "").strip()


def run_query(
    question: str,
    db: Path,
    top_k: int,
    show_sources: bool,
    retrieve_only: bool = False,
    *,
    fast: bool = True,
    conversation: list[dict] | None = None,
) -> dict:
    if retrieve_only:
        client = chromadb.PersistentClient(path=str(db))
        collection = client.get_collection("amharic_rag")
        embed_model = read_embed_model(db)
        env_m = os.environ.get("RAG_EMBED_MODEL", "").strip()
        if embed_model and env_m and env_m != embed_model:
            print(
                f"Warning: RAG_EMBED_MODEL={env_m!r} != indexed {embed_model!r}; using indexed model.",
                file=sys.stderr,
            )
        hits = retrieve(
            collection,
            retrieval_query_for(question, conversation),
            top_k=top_k,
            embed_model=embed_model,
            db=db,
        )
        return {
            "question": question,
            "answer": "",
            "retrieval_only": True,
            "retrieval": (
                "hybrid"
                if hits and hits[0].get("rrf_rank") is not None
                else "dense"
            ),
            "sources": [
                {
                    "source": h["meta"].get("source"),
                    "kind": h["meta"].get("kind"),
                    "page": h["meta"].get("page"),
                    "distance": h["distance"],
                    "text": h["text"],
                }
                for h in hits
            ],
        }

    pack = build_rag_pack(
        question, db, top_k, fast=fast, conversation=conversation
    )
    hits = pack["hits"]
    user_block = pack["user_block"]
    has_aux = bool(pack.get("has_aux_context"))

    use_ollama = os.environ.get("USE_OLLAMA", "1").strip() in ("1", "true", "yes")
    answer = ""

    if use_ollama:
        base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        model = default_chat_model(fast)
        opts = ollama_options(fast)
        system = system_for_rag(
            fast,
            has_aux_context=has_aux,
            follow_up=bool(conversation),
        )
        timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
        msgs = _ollama_messages(system, conversation, user_block)
        try:
            answer = ollama_chat_messages(
                msgs, model, base, options=opts, timeout_sec=timeout
            )
        except RuntimeError as e:
            answer = f"[Ollama]\n{e}"
    else:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not key:
            answer = (
                "ምንም LLM አልተዋቀረም። Ollama ለመጠቀም USE_OLLAMA=1 ያድርጉ ወይም "
                "USE_OLLAMA=0 እና OPENAI_API_KEY ያስገቡ።"
            )
        else:
            answer = openai_compatible_chat(base, key, model, user_block)

    out: dict = {
        "question": question,
        "answer": answer,
        "retrieval": pack["retrieval"],
        "tool_trace": pack.get("tool_trace") or [],
    }
    rq_used = (pack.get("retrieval_query") or "").strip()
    if rq_used and rq_used != question.strip():
        out["retrieval_query_used"] = rq_used[:1200]
    if show_sources:
        out["sources"] = format_source_rows(hits)
    return out


def stream_rag_answer(
    question: str,
    db: Path,
    top_k: int,
    *,
    fast: bool = True,
    conversation: list[dict] | None = None,
):
    """Sync retrieval, then stream LLM tokens (Ollama) or one chunk (OpenAI path).

    Returns (source preview rows, retrieval mode, generator factory).
    Call ``gen_fn()`` to obtain the iterator passed to ``st.write_stream``.
    """
    pack = build_rag_pack(
        question, db, top_k, fast=fast, conversation=conversation
    )
    sources = format_source_rows(pack["hits"])
    user_block = pack["user_block"]
    has_aux = bool(pack.get("has_aux_context"))

    def _gen():
        use_ollama = os.environ.get("USE_OLLAMA", "1").strip() in ("1", "true", "yes")
        system = system_for_rag(
            fast,
            has_aux_context=has_aux,
            follow_up=bool(conversation),
        )
        if not use_ollama:
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
            key = os.environ.get("OPENAI_API_KEY", "")
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            if not key:
                yield (
                    "ምንም LLM አልተዋቀረም። Ollama ለመጠቀም USE_OLLAMA=1 ያድርጉ ወይም "
                    "USE_OLLAMA=0 እና OPENAI_API_KEY ያስገቡ።"
                )
                return
            yield openai_compatible_chat(base, key, model, user_block)
            return
        base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        model = default_chat_model(fast)
        opts = ollama_options(fast)
        timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
        msgs = _ollama_messages(system, conversation, user_block)
        try:
            yield from iter_ollama_chat(
                msgs, model, base, options=opts, timeout_sec=timeout
            )
        except (RuntimeError, httpx.HTTPError, httpx.RequestError) as e:
            yield f"[Ollama]\n{e}"

    return sources, pack["retrieval"], _gen


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Amharic RAG query (default: fast ~3s — use --quality for slower, richer answers)"
    )
    ap.add_argument("question", nargs="?", help="Question in Amharic (or use --interactive)")
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent / "chroma_amharic",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval chunks (default: 4 fast / 6 quality; hybrid RRF uses larger pools)",
    )
    ap.add_argument(
        "--quality",
        action="store_true",
        help="Slower preset: more context + larger default model (qwen3:4b)",
    )
    ap.add_argument("--sources", action="store_true")
    ap.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Return top chunks only (no LLM; no Ollama/API needed)",
    )
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument(
        "--print-tools",
        action="store_true",
        help="Print Ollama-style JSON tool schemas (web_search, weather_forecast) and exit",
    )
    args = ap.parse_args()

    if args.print_tools:
        from rag_tools.registry import ollama_style_schemas

        print(json.dumps(ollama_style_schemas(), ensure_ascii=False, indent=2))
        return

    fast = _fast_mode(args.quality)
    top_k = args.top_k if args.top_k is not None else default_top_k(fast)

    if args.interactive:
        mode = "fast (~3s target)" if fast else "quality"
        print(f"አማርኛ RAG ({mode}) — መውጫ: quit ወይም exit")
        while True:
            try:
                q = input("\nጥያቄ> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in ("exit", "quit", "ዝውውር"):
                break
            res = run_query(
                q, args.db, top_k, args.sources, args.retrieve_only, fast=fast
            )
            if args.retrieve_only:
                print(json.dumps(res, ensure_ascii=False, indent=2))
            else:
                print("\nመልስ:\n", res["answer"])
                if args.sources and res.get("sources"):
                    print("\nምንጮች:\n", json.dumps(res["sources"], ensure_ascii=False, indent=2))
        return

    if not args.question:
        ap.error("pass a question or use --interactive")
    res = run_query(
        args.question, args.db, top_k, args.sources, args.retrieve_only, fast=fast
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
