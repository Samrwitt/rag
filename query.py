"""Query Amharic RAG: retrieve from Chroma + answer via Gemini, Groq, Ollama (Qwen), or OpenAI.

Retrieval: dense embeddings (Chroma) + optional BM25 sidecar with RRF fusion (see hybrid_retrieval).
LLM routing: ``RAG_LLM_BACKEND`` (auto|gemini|groq|ollama|openai); keys from ``.env`` (see llm_providers).
Auto prefers paid Gemini when ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` / ``GENAI_API_KEY`` exists,
then falls back to free Groq if ``GROQ_API_KEY`` exists and Gemini has a transient/quota/server error
(``GEMINI_GROQ_FALLBACK=0`` disables). Explicit Groq mode can still fall back to Gemini.
Hosted prompts are capped (``RAG_HOSTED_MESSAGES_MAX_CHARS``, default 26000) to reduce Groq 413.
Optional web/tools: rag_tools/augment.py. Optional dynamic_layer JSONL via ingest.
System prompt adds grounded recommendations and cautious predictions; disable with ``RAG_ADVISOR_PLAYBOOK=0``.
Hosted chat sends only the last ``RAG_HOSTED_CHAT_ROUNDS`` Q/A pairs (default 3) to save API tokens; ``0`` = unlimited.
When Groq/Gemini/OpenAI fail (e.g. 429), local Ollama is used if ``RAG_HOSTED_FALLBACK_OLLAMA=1`` (default) and ``USE_OLLAMA=1``.
Set ``RAG_LOCAL_FIRST=1`` for auto routing to Ollama before cloud APIs.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import chromadb
import httpx

from embeddings import encode_query
from llm_providers import (
    effective_llm_backend,
    gemini_chat_messages_with_groq_fallback,
    groq_chat_messages_with_gemini_fallback,
    iter_groq_chat_with_gemini_fallback,
    load_dotenv_if_present,
    openai_style_chat,
)
from nlu_farmer import (
    FarmerNLU,
    augment_retrieval_query_with_nlu,
    nlu_answer_scope_hint,
    parse_farmer_nlu,
)
from qa_context import qa_context_text
from rag_tools import augment_kb_context

load_dotenv_if_present()

# Short system line in fast mode saves prompt tokens / latency
SYSTEM_AM = (
    "አንተ የኢትዮጵያ ግብርና እና የተፈጥሮ ሀብት ባለሙያ ደረጃ ያለው የAI አማካሪ ነህ። "
    "ሁልጊዜ በአማርኛ መልስ፤ ተጠቃሚው በእንግሊዝኛ ቢጠይቅም መልሱ በአማርኛ ይሁን። "
    "የተሰጠህን መረጃ በቅድሚያ በመጠቀም ግልጽ፣ ትክክለኛ፣ ተግባራዊ እና እንደ GPT ደረጃ የተደራጀ መልስ ስጥ። "
    "ጥያቄውን እንደ መልስ አታድግም፤ ከመረጃው ቁልፍ ቁጥሮችንና እውቀትን አውጣ። "
    "መልስህ ለተጠቃሚ ቀጥታ የሚነገር ነገር ብቻ ይሁን። የፋይል ስም፣ «ምንጭ፡» ወይም የዚህ ቻት መመሪያ መስመር በመልስ ውስጥ አትጨምር። "
    "ከመረጃው ውጭ ከሆነ በግልጽ 'በዚህ መረጃ ውስጥ መልስ አልተገኘም' በማለት ግልጽ አድርግ።"
)
SYSTEM_AM_FAST = (
    "ሁልጊዜ በአማርኛ ቀጥተኛ መልስ ስጥ። የተሰጠውን መረጃ በቅድሚያ ተጠቀም። ከውጭ ከሆነ በግልጽ ብቻ ንገር። "
    "ጥያቄውን እንደ መልስ አታድግም፤ ከመረጃው ቁልፍ ቁጥሮችንና እውቀትን በግልጽ አውጣ። "
    "መልስህ ለገበሬው ቀጥታ የሚነገር ነገር ብቻ ይሁን። የፋይል ስም፣ «ምንጭ፡» ወይም የዚህ ቻት መመሪያ መስመር በመልስ ውስጥ አትጨምር። "
    "የሰንጠረይ፣ |፣ ^፣ [[፣ ወይም ረዥም የቁጥር ዝርዝሮችን አትጻፍ።"
)
SYSTEM_AUX_WEB = (
    " ከ[1] የሚጀመሩ ክፍሎች ከየእኛ መመሪያ ቤት ናቸው። [W1] … ከድር ወይም ከመሳሪያ ሲሆኑ "
    "አስቀድመህ የመመሪያውን መልስ አረጋግጥ፤ የድርን መረጃ በጥንቃቄ ተጠቀም።"
)
# Scope: same topic only; do not dump other crops. Must still extract facts, not echo the question.
STAY_ON_TOPIC_AM = (
    " ስለ ጥያቄው ርዕስ ብቻ መልስ። ከመመሪያው ሌሎች ሰብሎች ወይም ርዕሶች ካልተጠየቁ አትጨምር። "
    "ተጨማሪ ከሆነ በተመሳሳይ ርዕስ ላይ ብቻ በአጭር ይሁን። ጥያቄውን እንደ መልስ አታድግም።"
)
# Direct voice + one suggested next question (not the same as multi-turn «follow_up» in system_for_rag).
DIRECT_ANSWER_AND_FOLLOWUP_AM = (
    " መልስህን በቀጥታ እውቀት ጀምር። «መረጃው እንደሚለው»፣ «መረጃው»፣ «ከላይ»፣ «በመመሪያው» "
    "የመጀመሪያ መስመር አትጨምር። "
    "የመልስ ክፍልዎን በማጠናቀቅ ከመልስ በኋላ ባዶ መስመር አድርገው አንድ ብቻ ተዛማጅ ተጨማሪ ጥያቄ በአማርኛ ጻፍ "
    "(ለምሳሌ ስለ ዝናብ፣ ዝርያ፣ ወይም ማዳበሪያ)።"
)
# Grounded advisor: recommendations + cautious predictions (disable with RAG_ADVISOR_PLAYBOOK=0).
ADVISOR_PLAYBOOK_AM = (
    " ለገበሬው እንደ ብቁ አማካሪ አስብ፦ በመጀመሪያ ከላይ ካለው መመሪያ ውስጥ ያለውን አውጣ፤ ከዚያ "
    "በተግባር ሊሰሩ የሚችሉ ምክሮችና ቅድሚያ የሚሰጡ እርምጃዎች በአጭር ዝርዝር ስጥ። "
    "የወደፊት ውጤት፣ የበለጠ መልካም መስፈን ወይም ስጋት (prediction) ከመረጃው ሲደገፍ በግልጽ ተናገር፤ "
    "አካባቢ፣ ዝናብ፣ መሬት ወይም ዝርያ ካልታወቀ በግልጽ «ይህ በተለዋዋጭ ሁኔታ ላይ የተመረኰተ ግምት ነው» በማለት አሳስብ። "
    "ከመመሪያው ውጭ ያለውን እንደ ተረጋገጠ እውቀት አታقدር። "
    "ለመርዝ፣ ለዕጣዕጅ መድሃኒት፣ ለእንስሳ ጤና ቀውስ፣ ለሕጋዊ ወይም ለገንዘብ ውሳኔ ከአካባቢ ማራዝሚያ ወይም ባለሙያ "
    "እንዲጠየቅ በአንድ አጭር ሐረግ አስታውቅ።"
)
ADVISOR_PLAYBOOK_AM_FAST = (
    " ከመረጃው ጀምሮ ቀጥታ ምክርና እርምጃዎች ስጥ። ትንቢት/ግምት ከመረጃው በላይ ከሆነ በግልጽ እንደ ግምት አሳስብ። "
    "ለመርዝ/ዕጣዕጅ/እንስሳ ቀውስ ከባለሙያ ያማካኙ።"
)


def system_for_rag(
    fast: bool,
    *,
    has_aux_context: bool,
    follow_up: bool = False,
    nlu: FarmerNLU | None = None,
) -> str:
    base = (SYSTEM_AM_FAST if fast else SYSTEM_AM) + STAY_ON_TOPIC_AM + DIRECT_ANSWER_AND_FOLLOWUP_AM
    if os.environ.get("RAG_ADVISOR_PLAYBOOK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    ):
        base += ADVISOR_PLAYBOOK_AM_FAST if fast else ADVISOR_PLAYBOOK_AM
    base = base + nlu_answer_scope_hint(nlu)
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
    # Default to Qwen when using local Ollama (override with OLLAMA_MODEL).
    return "qwen2.5:3b" if fast else "qwen3:4b-instruct"


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
            "temperature": 0.06,
            "top_p": 0.82,
            "repeat_penalty": 1.12,
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "3072")),
            "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "240")),
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


def build_context(chunks: list[dict], max_chars: int, *, compact: bool = False) -> str:
    """Chunk headers use [n] only — filenames live in metadata / UI, not in LLM-visible text."""
    parts: list[str] = []
    n = 0
    for i, c in enumerate(chunks, 1):
        meta = c["meta"]
        kind = meta.get("kind", "")
        page = meta.get("page", "")
        head = f"[{i}]"
        if kind == "pdf" and page and not compact:
            head += f" ገጽ {page}"
        block = f"{head}\n{c['text']}\n"
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n".join(parts).strip()


def _strip_answer_meta_openers(text: str) -> str:
    """Drop leading phrases like «መረጃው እንደሚለው …»."""
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].strip()
    first = re.sub(
        r"^(?:መረጃው\s*እንደሚለው|መረጃው\s+እንደሚለው|እንደ\s*መረጃው|በመረጃው\s+መሰረት)\s*[፦:.\s]*",
        "",
        first,
        flags=re.IGNORECASE,
    ).lstrip()
    lines[0] = first
    return "\n".join(lines).strip()


def sanitize_chat_answer(text: str | None) -> str:
    """Remove «ምንጭ: file» lines, meta openers, and echoed RAG instructions from model output."""
    if not text or not str(text).strip():
        return (text or "").strip()
    t0 = text.strip()
    if re.match(r"^\[(?:groq|gemini|ollama|openai)\]", t0, re.I) or t0.startswith("[Ollama]"):
        return t0
    t0 = _strip_answer_meta_openers(t0)
    out: list[str] = []
    for line in t0.splitlines():
        s = line.strip()
        if re.match(r"^ምንጭ\s*[:፦]", s):
            continue
        if "ጥያቄውን እንደ መልስ አታድግም" in s:
            continue
        if "ከመረጃው ቁልፍ" in s and ("አውጣ" in s or "በግልጽ" in s):
            continue
        if "ከላይ ካለው መረጃ ቁጥር" in s and ("አውጣ" in s or "አታድግም" in s):
            continue
        out.append(line)
    joined = "\n".join(out).strip()
    joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
    return joined


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


_TOKEN_RE = re.compile(r"[\w\u1200-\u137F]{2,}")


def question_overlap_tokens(question: str) -> list[str]:
    """Amharic / word-ish tokens (length ≥2) for lexical reranking."""
    toks = _TOKEN_RE.findall(question or "")
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:28]


def rerank_hits_by_question_overlap(query_for_overlap: str, hits: list[dict]) -> list[dict]:
    """Prefer chunks whose text (or QA question) shares tokens with the overlap string."""
    if not hits:
        return hits
    if os.environ.get("RAG_RERANK_Q_OVERLAP", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    toks = question_overlap_tokens(query_for_overlap)
    if len(toks) < 2:
        return hits

    def key_pair(idx_h: tuple[int, dict]) -> tuple:
        idx, h = idx_h
        meta = h.get("meta") or {}
        blob = (h.get("text") or "") + " " + str(meta.get("question") or "")
        overlap = sum(1 for t in toks if t in blob)
        rrf = h.get("rrf_rank")
        try:
            rr = int(rrf) if rrf is not None else 9999
        except (TypeError, ValueError):
            rr = 9999
        dist = h.get("distance")
        try:
            d = float(dist) if dist is not None else 1e9
        except (TypeError, ValueError):
            d = 1e9
        return (-overlap, rr, d, idx)

    indexed = list(enumerate(hits))
    indexed.sort(key=key_pair)
    return [h for _, h in indexed]


# Crop rules: triggers (in user question), needles (must appear in chunk to be «about» that crop),
# rivals (other crops — demoted when the question names a crop but the chunk is only about a rival).
_CROP_TOPIC_RULES: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "id": "coffee",
        "triggers": ("ለቡና", "ቡና", "ቡናን"),
        "needles": (
            "ቡና",
            "ለቡና",
            "ጎማ",
            "coffee",
            "አረቢካ",
            "አረብካ",
            "ሮቡስታ",
            "አራቢካ",
        ),
        "rivals": ("ስንዴ", "ለስንዴ", "wheat", "ሰሊጥ", "ሰሊት", "ገብስ", "ቴፍ", "teff"),
    },
    {
        "id": "wheat",
        "triggers": ("ስንዴ",),
        "needles": ("ስንዴ", "wheat"),
        "rivals": ("ቡና", "ለቡና", "coffee", "ሰሊጥ", "ገብስ"),
    },
    {
        "id": "sesame",
        "triggers": ("ሰሊጥ", "ሰሊት"),
        "needles": ("ሰሊጥ", "ሰሊት", "sesame"),
        "rivals": ("ስንዴ", "ቡና", "ለቡና", "ገብስ"),
    },
    {
        "id": "barley",
        "triggers": ("ገብስ",),
        "needles": ("ገብስ", "barley"),
        "rivals": ("ስንዴ", "ቡና", "ለቡና"),
    },
    {
        "id": "faba",
        "triggers": ("ቡቃያ",),
        "needles": ("ቡቃያ",),
        "rivals": ("ስንዴ", "ቡና", "ለቡና"),
    },
    {
        "id": "potato",
        "triggers": ("ድንች",),
        "needles": ("ድንች", "potato"),
        "rivals": ("ስንዴ", "ቡና", "ለቡና"),
    },
)


def _match_crop_topic_rule(question: str) -> dict[str, tuple[str, ...]] | None:
    q = question or ""
    for row in _CROP_TOPIC_RULES:
        if any(t in q for t in row["triggers"]):
            return row
    return None


def _crop_rule_by_id(crop_id: str | None) -> dict[str, tuple[str, ...]] | None:
    if not crop_id:
        return None
    for row in _CROP_TOPIC_RULES:
        if row.get("id") == crop_id:
            return row
    return None


def effective_crop_topic_rule(question: str, nlu: FarmerNLU) -> dict[str, tuple[str, ...]] | None:
    """Prefer NLU crop id, else substring triggers in the raw question."""
    r = _crop_rule_by_id(nlu.crop_id)
    if r:
        return r
    return _match_crop_topic_rule(question)


def _crop_row_matches_question(
    row: dict[str, tuple[str, ...]],
    question: str,
    nlu: FarmerNLU,
) -> bool:
    if nlu.crop_id and row.get("id") == nlu.crop_id:
        return True
    return any(t in question for t in row["triggers"])


def _retrieval_blob(h: dict) -> str:
    m = h.get("meta") or {}
    return ((h.get("text") or "") + " " + str(m.get("question") or "")).strip()


def _dedupe_hits_preserve_order(hits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        m = h.get("meta") or {}
        key = f"{m.get('source', '')}\0{m.get('page', '')}\0{(h.get('text') or '')[:240]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _topic_good_count(question: str, hits: list[dict], nlu: FarmerNLU) -> int:
    row = effective_crop_topic_rule(question, nlu)
    if not row:
        return len(hits)
    needles = row["needles"]
    return sum(1 for h in hits if any(n in _retrieval_blob(h) for n in needles))


def filter_cross_crop_hits(question: str, hits: list[dict], nlu: FarmerNLU) -> list[dict]:
    """Demote chunks about a rival crop when the question names a specific crop (local KB only)."""
    if not hits or os.environ.get("RAG_TOPIC_FILTER", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    row = effective_crop_topic_rule(question, nlu)
    if not row:
        return hits
    needles, rivals = row["needles"], row["rivals"]
    good: list[dict] = []
    neutral: list[dict] = []
    bad: list[dict] = []
    for h in hits:
        b = _retrieval_blob(h)
        has_needle = any(n in b for n in needles)
        has_rival = any(r in b for r in rivals)
        if has_needle:
            good.append(h)
        elif has_rival:
            bad.append(h)
        else:
            neutral.append(h)
    if good:
        return good + neutral + bad
    return neutral + bad


def topic_vector_refine_retrieve(
    rq_boost: str,
    collection,
    embed_model: str | None,
    db: Path | None,
    pool_k: int,
    hits: list[dict],
) -> list[dict]:
    """Second Chroma vector/hybrid pass with an embedding query augmented by crop keywords."""
    if os.environ.get("RAG_TOPIC_VECTOR_REFINE", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    try:
        extra = retrieve(collection, rq_boost, top_k=pool_k, embed_model=embed_model, db=db)
    except Exception as e:
        print(f"Warning: topic vector refine retrieve failed: {e}", file=sys.stderr)
        return hits
    return _dedupe_hits_preserve_order(extra + hits)


def boost_hits_by_topic_keywords(
    question: str, hits: list[dict], nlu: FarmerNLU
) -> list[dict]:
    """Move crop/topic-specific chunks first when the question names a crop (Amharic)."""
    if not hits or os.environ.get("RAG_TOPIC_BOOST", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return hits
    q = question or ""
    for row in _CROP_TOPIC_RULES:
        if not _crop_row_matches_question(row, q, nlu):
            continue
        needles = row["needles"]
        on_topic: list[dict] = []
        rest: list[dict] = []
        for h in hits:
            blob = _retrieval_blob(h)
            if any(n in blob for n in needles):
                on_topic.append(h)
            else:
                rest.append(h)
        if on_topic:
            return on_topic + rest
    return hits


def retrieve_ranked_hits(
    question: str,
    collection,
    db: Path,
    top_k: int,
    embed_model: str | None,
    conversation: list[dict] | None,
    nlu: FarmerNLU,
) -> tuple[list[dict], str]:
    """Chroma vector/hybrid retrieval + rerank + crop filter; optional second pass if pool has no crop match."""
    rq = augment_retrieval_query_with_nlu(
        retrieval_query_for(question, conversation), nlu
    )
    mult = max(1, int(os.environ.get("RAG_RETRIEVE_POOL_MULT", "4")))
    cap = max(8, int(os.environ.get("RAG_RETRIEVE_POOL_MAX", "48")))
    pool_k = min(cap, max(top_k, top_k * mult))

    hits = retrieve(collection, rq, top_k=pool_k, embed_model=embed_model, db=db)
    hits = rerank_hits_by_question_overlap(rq, hits)
    hits = boost_hits_by_topic_keywords(question, hits, nlu)
    hits = filter_cross_crop_hits(question, hits, nlu)
    row = effective_crop_topic_rule(question, nlu)
    if row and _topic_good_count(question, hits, nlu) == 0:
        tail_parts = list(row["needles"][: min(5, len(row["needles"]))])
        if nlu.aspect == "altitude" or "ከፍታ" in question:
            tail_parts.append("ከፍታ")
        if nlu.aspect == "price":
            tail_parts.extend(["ዋጋ", "ገበያ"])
        if nlu.aspect == "rainfall":
            tail_parts.append("ዝናብ")
        if nlu.aspect == "soil":
            tail_parts.append("አፈር")
        if nlu.aspect == "fertilizer":
            tail_parts.append("ማዳበሪያ")
        tail = " ".join(dict.fromkeys(tail_parts))
        rq_boost = f"{rq.strip()}\n{tail}".strip()
        hits = topic_vector_refine_retrieve(
            rq_boost, collection, embed_model, db, pool_k, hits
        )
        hits = rerank_hits_by_question_overlap(rq_boost, hits)
        hits = boost_hits_by_topic_keywords(question, hits, nlu)
        hits = filter_cross_crop_hits(question, hits, nlu)
    hits = hits[:top_k]
    return hits, rq


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
    nlu = parse_farmer_nlu(question)
    hits, rq = retrieve_ranked_hits(
        question, collection, db, top_k, embed_model, conversation, nlu
    )
    extra_ctx, tool_trace = augment_kb_context(question, hits, fast=fast)
    base_max = int(os.environ.get("RAG_CONTEXT_CHARS", "4200" if fast else "9000"))
    reserved = min(len(extra_ctx) + 400, 3600) if extra_ctx.strip() else 0
    max_chars = max(900, base_max - reserved)
    ctx = build_context(hits, max_chars=max_chars, compact=fast)
    has_aux = bool(extra_ctx.strip())
    aux_block = f"ተጨማሪ (ድር / መሳሪያ — ከመመሪያ ቤት ይለያል፤ [W1] …)፦\n{extra_ctx}\n\n" if has_aux else ""
    # Keep user message short: long Amharic tails were echoed by the model as «answers».
    user_block = (
        f"ጥያቄ፦ {question.strip()}\n\n"
        f"መረጃ፦\n{ctx}\n\n"
        f"{aux_block}"
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
        "nlu": {
            "crop_id": nlu.crop_id,
            "aspect": nlu.aspect,
            "retrieval_boost": nlu.retrieval_boost or None,
        },
    }


def _hosted_chat_rounds_limit() -> int:
    """Max prior user/assistant *pairs* sent to Groq/Gemini/OpenAI. ``0`` / ``off`` = unlimited."""
    v = os.environ.get("RAG_HOSTED_CHAT_ROUNDS", "3").strip().lower()
    if not v or v in ("0", "off", "unlimited", "no", "false"):
        return 0
    try:
        return max(0, int(v))
    except ValueError:
        return 3


def trim_hosted_conversation_messages(messages: list[dict]) -> list[dict]:
    """Keep system + last N Q/A turns + final RAG user message (reduces token quota burn)."""
    if len(messages) <= 2:
        return messages
    if (messages[0].get("role") or "") != "system":
        return messages
    limit = _hosted_chat_rounds_limit()
    if limit == 0:
        return messages
    system = messages[0]
    tail = messages[-1]
    mid = messages[1:-1]
    max_mid = limit * 2
    if len(mid) <= max_mid:
        return messages
    return [system] + mid[-max_mid:] + [tail]


def _hosted_ollama_fallback_enabled() -> bool:
    if os.environ.get("RAG_HOSTED_FALLBACK_OLLAMA", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    return os.environ.get("USE_OLLAMA", "1").strip().lower() in ("1", "true", "yes")


def _ollama_failover_answer(msgs: list[dict], fast: bool) -> str:
    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = default_chat_model(fast)
    opts = ollama_options(fast)
    timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
    return ollama_chat_messages(msgs, model, base, options=opts, timeout_sec=timeout)


def _prepare_llm_messages(
    system: str,
    conversation: list[dict] | None,
    user_block: str,
    backend: str,
) -> list[dict]:
    msgs = _ollama_messages(system, conversation, user_block)
    if backend in ("groq", "gemini", "openai"):
        msgs = trim_hosted_conversation_messages(msgs)
        msgs = shrink_messages_for_hosted_api(msgs)
    return msgs


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


def shrink_messages_for_hosted_api(messages: list[dict]) -> list[dict]:
    """Cap total characters to reduce Groq/OpenAI 413 (payload too large) on long RAG + chat."""
    cap = max(12_000, int(os.environ.get("RAG_HOSTED_MESSAGES_MAX_CHARS", "26000")))
    mark = "\n\n…(ለ API መጠን ተሰነዘለ)…"
    out: list[dict] = [{"role": m["role"], "content": str(m.get("content") or "")} for m in messages]

    def total() -> int:
        return sum(len(x["content"]) for x in out)

    for _ in range(32):
        if total() <= cap:
            return out
        over = total() - cap + len(mark) + 40
        cut_idx: int | None = None
        for idx in range(len(out) - 1, -1, -1):
            if out[idx]["role"] not in ("user", "assistant"):
                continue
            c = out[idx]["content"]
            if len(c) < 2800:
                continue
            cut_idx = idx
            break
        if cut_idx is not None:
            c = out[cut_idx]["content"]
            new_len = max(2500, len(c) - max(over, int(0.12 * len(c))))
            out[cut_idx]["content"] = c[:new_len].rstrip() + mark
            continue
        # Short users only: trim longest message (often system or last user)
        li = max(range(len(out)), key=lambda i: len(out[i]["content"]))
        c = out[li]["content"]
        if len(c) <= 1800:
            break
        new_len = max(1500, len(c) - over)
        out[li]["content"] = c[:new_len].rstrip() + mark
    return out


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
        nlu = parse_farmer_nlu(question)
        hits, rq = retrieve_ranked_hits(
            question, collection, db, top_k, embed_model, conversation, nlu
        )
        out = {
            "question": question,
            "answer": "",
            "retrieval_only": True,
            "nlu": {
                "crop_id": nlu.crop_id,
                "aspect": nlu.aspect,
                "retrieval_boost": nlu.retrieval_boost or None,
            },
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
        if rq.strip() != question.strip():
            out["retrieval_query_used"] = rq[:1200]
        return out

    pack = build_rag_pack(
        question, db, top_k, fast=fast, conversation=conversation
    )
    hits = pack["hits"]
    user_block = pack["user_block"]
    has_aux = bool(pack.get("has_aux_context"))

    system = system_for_rag(
        fast,
        has_aux_context=has_aux,
        follow_up=bool(conversation),
        nlu=parse_farmer_nlu(question),
    )
    backend = effective_llm_backend()
    msgs = _prepare_llm_messages(system, conversation, user_block, backend)
    answer = ""
    llm_used = backend
    hosted_timeout = float(os.environ.get("RAG_HOSTED_HTTP_TIMEOUT", "120" if fast else "240"))

    try:
        if backend == "groq":
            answer, llm_used = groq_chat_messages_with_gemini_fallback(
                msgs, fast=fast, timeout_sec=hosted_timeout
            )
        elif backend == "gemini":
            answer, llm_used = gemini_chat_messages_with_groq_fallback(
                msgs, fast=fast, timeout_sec=hosted_timeout
            )
        elif backend == "openai":
            base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            if not key:
                answer = "OPENAI_API_KEY አልተዋቀረም። RAG_LLM_BACKEND=groq ወይም gemini ይሞክሩ።"
            else:
                answer = openai_style_chat(
                    msgs,
                    base_url=base,
                    api_key=key,
                    model=model,
                    timeout_sec=hosted_timeout,
                )
        else:
            # ollama (local Qwen / etc.)
            if os.environ.get("USE_OLLAMA", "1").strip() not in ("1", "true", "yes"):
                answer = (
                    "USE_OLLAMA=0 ነው። RAG_LLM_BACKEND=groq ወይም gemini ይመርጡ ወይም USE_OLLAMA=1 ያድርጉ።"
                )
            else:
                base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                model = default_chat_model(fast)
                opts = ollama_options(fast)
                timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
                try:
                    answer = ollama_chat_messages(
                        msgs, model, base, options=opts, timeout_sec=timeout
                    )
                except RuntimeError as e:
                    answer = f"[Ollama]\n{e}"
    except Exception as e:
        if backend in ("groq", "gemini", "openai") and _hosted_ollama_fallback_enabled():
            try:
                answer = _ollama_failover_answer(msgs, fast)
                llm_used = "ollama"
            except Exception as fe:
                answer = f"[{backend}]\n{e}\n[Ollama]\n{fe}"
        else:
            answer = f"[{backend}]\n{e}"

    answer = sanitize_chat_answer(answer)

    out: dict = {
        "question": question,
        "answer": answer,
        "retrieval": pack["retrieval"],
        "tool_trace": pack.get("tool_trace") or [],
        "llm_backend": llm_used,
        "nlu": pack.get("nlu"),
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
        system = system_for_rag(
            fast,
            has_aux_context=has_aux,
            follow_up=bool(conversation),
            nlu=parse_farmer_nlu(question),
        )
        backend = effective_llm_backend()
        msgs = _prepare_llm_messages(system, conversation, user_block, backend)
        hosted_timeout = float(os.environ.get("RAG_HOSTED_HTTP_TIMEOUT", "120" if fast else "240"))
        try:
            if backend == "groq":
                yield from iter_groq_chat_with_gemini_fallback(
                    msgs, fast=fast, timeout_sec=hosted_timeout
                )
            elif backend == "gemini":
                answer, _used = gemini_chat_messages_with_groq_fallback(
                    msgs, fast=fast, timeout_sec=hosted_timeout
                )
                yield answer
            elif backend == "openai":
                base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
                key = os.environ.get("OPENAI_API_KEY", "").strip()
                model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
                if not key:
                    yield "OPENAI_API_KEY አልተዋቀረም።"
                    return
                yield openai_style_chat(
                    msgs,
                    base_url=base,
                    api_key=key,
                    model=model,
                    timeout_sec=hosted_timeout,
                )
            else:
                if os.environ.get("USE_OLLAMA", "1").strip() not in ("1", "true", "yes"):
                    yield "USE_OLLAMA=0 — RAG_LLM_BACKEND=groq ወይም gemini ይጠቀሙ።"
                    return
                base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                model = default_chat_model(fast)
                opts = ollama_options(fast)
                timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
                try:
                    yield from iter_ollama_chat(
                        msgs, model, base, options=opts, timeout_sec=timeout
                    )
                except (RuntimeError, httpx.HTTPError, httpx.RequestError) as e:
                    yield f"[Ollama]\n{e}"
        except Exception as e:
            if backend in ("groq", "gemini", "openai") and _hosted_ollama_fallback_enabled():
                try:
                    base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
                    model = default_chat_model(fast)
                    opts = ollama_options(fast)
                    timeout = float(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120" if fast else "300"))
                    yield from iter_ollama_chat(
                        msgs, model, base, options=opts, timeout_sec=timeout
                    )
                    return
                except Exception as fe:
                    yield f"[{backend}]\n{e}\n[Ollama]\n{fe}"
                    return
            yield f"[{backend}]\n{e}"

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
