#!/usr/bin/env python3
"""Lightweight eval harness: retrieval quality + optional end-to-end answer checks.

Examples::

    # Retrieval only (no Ollama): fast, good for CI
    python eval/run_eval.py --db ./chroma_amharic --no-llm

    # Full pipeline (needs Ollama or OPENAI)
    python eval/run_eval.py --db ./chroma_amharic --limit 3

Gold file: JSONL with objects::

    id: str
    question: str
    hits_must_contain_any: list[str]  # at least one substring appears in some hit text
    answer_keywords: list[str]        # each must appear in answer (if --with-llm)
    skip_retrieval: optional bool     # skip hit checks (e.g. !web smoke)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

# Project root on sys.path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query import build_rag_pack, default_top_k, run_query  # noqa: E402


def _load_gold(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _retrieval_pass(hits: list[dict], needles: list[str]) -> bool:
    if not needles:
        return True
    blob = "\n".join((h.get("text") or "") for h in hits)
    return any(n in blob for n in needles)


def _answer_pass(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    a = answer or ""
    return all(k in a for k in keywords)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG eval harness (retrieval + optional LLM)")
    ap.add_argument(
        "--db",
        type=Path,
        default=ROOT / "chroma_amharic",
        help="Chroma persist directory",
    )
    ap.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "eval" / "gold.jsonl",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max cases (0 = all)")
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Only retrieval checks via build_rag_pack (no Ollama/OpenAI)",
    )
    ap.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run full run_query and score answer_keywords",
    )
    ap.add_argument("--fast", action="store_true", help="Fast preset for LLM leg")
    ap.add_argument("--quality", action="store_true", help="Quality preset for LLM leg")
    args = ap.parse_args()

    fast = not args.quality
    if args.fast:
        fast = True

    gold_rows = _load_gold(args.gold)
    if args.limit:
        gold_rows = gold_rows[: args.limit]

    pack_times: list[float] = []
    llm_times: list[float] = []
    ret_scores: list[float] = []
    ans_scores: list[float] = []
    details: list[dict[str, Any]] = []

    for row in gold_rows:
        q = row["question"]
        t0 = time.perf_counter()
        pack = build_rag_pack(q, args.db, default_top_k(fast), fast=fast)
        pack_times.append(time.perf_counter() - t0)
        hits = pack["hits"]
        skip_r = bool(row.get("skip_retrieval"))
        needles = row.get("hits_must_contain_any") or []
        r_ok = True if skip_r else _retrieval_pass(hits, needles)
        ret_scores.append(1.0 if r_ok else 0.0)

        a_ok = True
        answer = ""
        if args.with_llm and not args.no_llm:
            t2 = time.perf_counter()
            out = run_query(
                q,
                args.db,
                default_top_k(fast),
                show_sources=False,
                fast=fast,
            )
            llm_times.append(time.perf_counter() - t2)
            answer = out.get("answer") or ""
            a_ok = _answer_pass(answer, row.get("answer_keywords") or [])
            ans_scores.append(1.0 if a_ok else 0.0)

        details.append(
            {
                "id": row.get("id"),
                "retrieval_pass": r_ok,
                "answer_pass": a_ok if args.with_llm and not args.no_llm else None,
                "tool_trace": pack.get("tool_trace"),
            }
        )

    n = len(gold_rows)
    summary = {
        "cases": n,
        "retrieval_accuracy": round(statistics.mean(ret_scores), 4) if ret_scores else None,
        "answer_accuracy": round(statistics.mean(ans_scores), 4)
        if ans_scores
        else None,
        "latency_build_pack_s_p50": round(statistics.median(pack_times), 4) if pack_times else None,
        "latency_llm_s_p50": round(statistics.median(llm_times), 4) if llm_times else None,
        "gold_path": str(args.gold),
        "db": str(args.db),
    }
    print(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
