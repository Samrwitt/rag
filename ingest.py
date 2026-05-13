"""Build Chroma index from KB/*.pdf, merged.json Q&A, and optional dynamic_layer chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from tqdm import tqdm

from chunking import chunk_text
from embeddings import encode_passages, env_model_name as model_name
from hybrid_retrieval import save_hybrid_sidecar
from qa_context import qa_index_body


def pdf_to_page_texts(path: Path) -> list[str]:
    doc = fitz.open(path)
    pages: list[str] = []
    try:
        for i in range(len(doc)):
            t = doc[i].get_text("text") or ""
            t = t.strip()
            if t:
                pages.append(t)
    finally:
        doc.close()
    return pages


def load_qa(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("merged.json must be a JSON array")
    return raw


def _scalar_meta(v) -> str | int | float | bool:
    if v is None:
        return ""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return v[:8192]
    return json.dumps(v, ensure_ascii=False)[:8192]


def load_dynamic_jsonl(path: Path) -> tuple[list[str], list[str], list[dict]]:
    """Load ``dynamic_layer.py`` output (one JSON object per line) for Chroma + BM25."""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    if not path.is_file():
        return ids, documents, metadatas
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skip dynamic line {line_no}: {e}")
                continue
            cid = (row.get("id") or "").strip() or f"line_{line_no}"
            body = (row.get("text_am") or row.get("text") or "").strip()
            if len(body) < 10:
                continue
            ids.append(f"dyn:{cid}")
            documents.append(body)
            kb = (row.get("kb") or "dynamic").strip()
            meta: dict = {
                "source": (row.get("source_url") or row.get("source_org") or cid)[:1024],
                "kind": kb,
                "page": (row.get("updated_at") or row.get("validity") or "")[:512],
                "data_layer": (row.get("data_layer") or "")[:256],
                "source_org": (row.get("source_org") or "")[:512],
                "dynamic_id": cid[:512],
            }
            for opt_key in (
                "location",
                "region",
                "latitude",
                "longitude",
                "requested_location",
                "update_frequency",
                "language_segment",
                "source_type",
            ):
                if opt_key in row and row[opt_key] is not None:
                    meta[opt_key] = _scalar_meta(row[opt_key])
            metadatas.append(meta)
    return ids, documents, metadatas


def main() -> None:
    root = Path(__file__).resolve().parent
    default_dynamic = root / "data" / "chunks" / "dynamic_context_chunks.jsonl"

    ap = argparse.ArgumentParser(description="Ingest PDFs + QA (+ optional dynamic_layer JSONL) into Chroma")
    ap.add_argument(
        "--kb",
        type=Path,
        default=root / "KB",
        help="Folder with PDF manuals",
    )
    ap.add_argument(
        "--qa",
        type=Path,
        default=root / "merged.json",
        help="JSON array of {q,d}",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=root / "chroma_amharic",
        help="Chroma persist directory",
    )
    ap.add_argument(
        "--dynamic-jsonl",
        type=Path,
        default=default_dynamic,
        help="dynamic_layer.py output; ingested when file exists (unless --no-dynamic)",
    )
    ap.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Do not merge dynamic JSONL even if the file exists",
    )
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    args = ap.parse_args()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    # --- QA pairs (high precision for matching questions) ---
    if args.qa.is_file():
        qa = load_qa(args.qa)
        for i, row in enumerate(qa):
            q = (row.get("q") or "").strip()
            d = (row.get("d") or "").strip()
            if not q and not d:
                continue
            body = qa_index_body(q, d)
            if len(body) < 15:
                continue
            ids.append(f"qa:{i}")
            documents.append(body)
            metadatas.append(
                {
                    "source": str(args.qa.name),
                    "kind": "qa",
                    "idx": str(i),
                    "question": q,
                    "answer": d,
                }
            )
    else:
        print(f"QA file not found: {args.qa}, skipping")

    # --- PDFs ---
    pdf_paths = sorted(args.kb.glob("*.pdf")) if args.kb.is_dir() else []
    pdf_chunk_i = 0
    for pdf in tqdm(pdf_paths, desc="PDFs"):
        try:
            page_texts = pdf_to_page_texts(pdf)
        except Exception as e:
            print(f"Skip {pdf.name}: {e}")
            continue
        for pi, page_t in enumerate(page_texts):
            for ci, ch in enumerate(chunk_text(page_t)):
                ids.append(f"pdf:{pdf.name}:{pi}:{ci}:{pdf_chunk_i}")
                pdf_chunk_i += 1
                documents.append(ch)
                metadatas.append(
                    {
                        "source": pdf.name,
                        "kind": "pdf",
                        "page": str(pi + 1),
                    }
                )

    # --- Dynamic layer (weather, market pointers, soil metadata, CIAT pages) ---
    if not args.no_dynamic and args.dynamic_jsonl.is_file():
        di, dd, dm = load_dynamic_jsonl(args.dynamic_jsonl)
        if di:
            ids.extend(di)
            documents.extend(dd)
            metadatas.extend(dm)
            print(f"Dynamic chunks from {args.dynamic_jsonl}: {len(di)}")
    elif not args.no_dynamic:
        print(f"No dynamic JSONL at {args.dynamic_jsonl} (run dynamic_layer.py first); skipping.")

    if not documents:
        raise SystemExit("No documents to index. Check KB/ and merged.json paths.")

    print(f"Total chunks: {len(documents)} | embed_model={model_name()}")

    client = chromadb.PersistentClient(path=str(args.db))
    col_name = "amharic_rag"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )
    args.db.mkdir(parents=True, exist_ok=True)
    (args.db / "embed_model.txt").write_text(model_name(), encoding="utf-8")

    for start in tqdm(range(0, len(documents), args.batch), desc="Embedding batches"):
        batch_ids = ids[start : start + args.batch]
        batch_docs = documents[start : start + args.batch]
        batch_meta = metadatas[start : start + args.batch]
        vecs = encode_passages(batch_docs, batch_size=min(args.batch, 32))
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=vecs.tolist(),
        )

    save_hybrid_sidecar(args.db, ids, documents)
    print(f"Done. Chroma + hybrid BM25 at {args.db} (re-run query after ingest; RAG_HYBRID=0 disables hybrid)")


if __name__ == "__main__":
    main()
