"""BM25 + dense Chroma retrieval fused with RRF (Reciprocal Rank Fusion)."""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from qa_context import qa_context_text

_bm_cache: dict[str, tuple[float, BM25Okapi, list[str]]] = {}

# Very generic tokens — down-weighted for lexical rescoring only (BM25 still uses full query)
_QUERY_STOP = frozenset(
    {
        "ምን",
        "ነው",
        "እንዴት",
        "ወይም",
        "በ",
        "የ",
        "ና",
        "ላይ",
        "ግን",
        "እስከ",
        "ያስፈልጋል",
        "ይፈልጋል",
    }
)


def tokenize(text: str) -> list[str]:
    """Amharic (Ethiopic) + Latin/alnum tokens for BM25."""
    text = (text or "").strip().lower()
    if not text:
        return []
    parts = re.findall(r"[\u1200-\u137F]+|[a-z0-9]+", text, re.I)
    return parts if parts else text.split()


def content_query_tokens(query: str) -> list[str]:
    return [t for t in tokenize(query) if len(t) >= 2 and t not in _QUERY_STOP]


def overlap_score(query: str, text: str) -> float:
    """Higher when chunk text hits more salient (non-stop) query tokens."""
    tl = (text or "").lower()
    s = 0.0
    for t in content_query_tokens(query):
        if t in tl:
            s += 1.0 + min(3.0, 0.08 * len(t))
            if len(t) >= 4:
                s += 1.5
    return s


def text_for_overlap_ranking(meta: dict, text: str) -> str:
    """For QA rows, rank by question only so answers do not swamp the user query."""
    m = meta or {}
    if m.get("kind") == "qa":
        qm = m.get("question")
        if isinstance(qm, str) and qm.strip():
            return qm.strip()
        if "ጥያቄ፦" in text and "ምላሽ፦" in text:
            return text.split("ምላሽ፦", 1)[0]
        if "\n\n" in text:
            head = text.split("\n\n", 1)[0].strip()
            if head:
                return head
    return text


def save_hybrid_sidecar(db: Path, ids: list[str], documents: list[str]) -> None:
    db.mkdir(parents=True, exist_ok=True)
    tokens = [tokenize(d) for d in documents]
    with open(db / "hybrid_bm25.pkl", "wb") as f:
        pickle.dump({"ids": ids, "tokens": tokens}, f, protocol=pickle.HIGHEST_PROTOCOL)
    key = str(db.resolve())
    _bm_cache.pop(key, None)


def _load_bm25(db: Path) -> tuple[BM25Okapi, list[str]] | None:
    p = db / "hybrid_bm25.pkl"
    if not p.is_file():
        return None
    key = str(db.resolve())
    mtime = p.stat().st_mtime
    hit = _bm_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1], hit[2]
    with open(p, "rb") as f:
        data = pickle.load(f)
    ids: list[str] = data["ids"]
    tok: list[list[str]] = data["tokens"]
    bm25 = BM25Okapi(tok)
    _bm_cache[key] = (mtime, bm25, ids)
    return bm25, ids


def sidecar_exists(db: Path) -> bool:
    return (db / "hybrid_bm25.pkl").is_file()


def rrf_fuse(rank_lists: list[list[str]], k_rrf: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for rlist in rank_lists:
        for rank, doc_id in enumerate(rlist):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def hybrid_retrieve(
    collection,
    db: Path,
    query: str,
    embed_query_vec: list[float],
    final_k: int,
    *,
    dense_pool: int | None = None,
    bm25_pool: int | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """Return same shape as dense-only rows in query.retrieve."""
    loaded = _load_bm25(db)
    if loaded is None:
        raise RuntimeError("hybrid sidecar missing")
    bm25, id_order = loaded

    try:
        n_docs = int(collection.count())
    except Exception:
        n_docs = 10_000_000

    dp = dense_pool or int(os.environ.get("RAG_DENSE_POOL", "56"))
    bp = bm25_pool or int(os.environ.get("RAG_BM25_POOL", "56"))
    rk = rrf_k or int(os.environ.get("RAG_RRF_K", "60"))

    dp = max(8, min(dp, max(n_docs, 1)))
    bp = max(8, min(bp, len(id_order)))

    res = collection.query(
        query_embeddings=[embed_query_vec],
        n_results=dp,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = (res.get("ids") or [[]])[0]
    dense_dists = (res.get("distances") or [[]])[0]
    dist_by_id = dict(zip(dense_ids, dense_dists))

    q_tok = tokenize(query)
    scores = bm25.get_scores(q_tok) if q_tok else [0.0] * len(id_order)
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bp]
    bm25_ids = [id_order[i] for i in bm25_order if i < len(id_order)]

    fused = rrf_fuse([dense_ids, bm25_ids], k_rrf=rk)
    rerank_pool = int(os.environ.get("RAG_RERANK_POOL", "24"))
    rerank_pool = max(final_k, min(rerank_pool, len(fused)))

    seen: set[str] = set()
    fused_unique: list[str] = []
    for fid in fused:
        if fid in seen:
            continue
        seen.add(fid)
        fused_unique.append(fid)
        if len(fused_unique) >= rerank_pool:
            break

    if not fused_unique:
        return []

    got = collection.get(
        ids=fused_unique,
        include=["documents", "metadatas"],
    )
    g_ids = got.get("ids") or []
    g_docs = got.get("documents") or []
    g_meta = got.get("metadatas") or []
    by_id: dict[str, dict[str, Any]] = {}
    for i, gid in enumerate(g_ids):
        by_id[gid] = {
            "text": g_docs[i] if i < len(g_docs) else "",
            "meta": g_meta[i] if i < len(g_meta) else {},
        }

    rows: list[dict] = []
    for rank, fid in enumerate(fused_unique):
        cell = by_id.get(fid)
        if not cell:
            continue
        raw = cell["text"]
        rows.append(
            {
                "text": raw,
                "meta": cell["meta"] or {},
                "distance": dist_by_id.get(fid),
                "rrf_rank": rank,
            }
        )

    rows.sort(
        key=lambda r: (
            -overlap_score(
                query,
                text_for_overlap_ranking(r["meta"], r["text"]),
            ),
            r.get("rrf_rank", 99),
        )
    )
    rows = rows[:final_k]
    for i, r in enumerate(rows):
        r["text"] = qa_context_text(r["meta"], r["text"])
        r["rrf_rank"] = i
    return rows
