#!/usr/bin/env python3
"""Build hybrid_bm25.pkl from an existing Chroma DB (no re-embedding).

Use when you already ingested before hybrid was added:
  cd /home/lenovo/Desktop/RAG && . .venv/bin/activate && python build_hybrid_sidecar_only.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from hybrid_retrieval import save_hybrid_sidecar


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BM25 sidecar from existing Chroma only")
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parent / "chroma_amharic",
    )
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=str(args.db))
    col = client.get_collection("amharic_rag")

    all_ids: list[str] = []
    all_docs: list[str] = []
    offset = 0
    while True:
        g = col.get(
            limit=args.batch,
            offset=offset,
            include=["documents"],
        )
        ids = g.get("ids") or []
        docs = g.get("documents") or []
        if not ids:
            break
        for i, doc in zip(ids, docs):
            all_ids.append(i)
            all_docs.append((doc or "").strip() or " ")
        offset += len(ids)
        if len(ids) < args.batch:
            break

    if not all_ids:
        raise SystemExit("No documents in Chroma collection.")

    save_hybrid_sidecar(args.db, all_ids, all_docs)
    print(f"Wrote hybrid_bm25.pkl for {len(all_ids)} chunks under {args.db}")


if __name__ == "__main__":
    main()
