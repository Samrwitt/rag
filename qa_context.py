"""QA chunk text: index form vs context shown to the LLM."""

from __future__ import annotations


def qa_index_body(question: str, answer: str) -> str:
    """Stored in Chroma + BM25: question + answer so retrieval matches user questions."""
    q = (question or "").strip()
    d = (answer or "").strip()
    if q and d:
        return f"{q}\n\n{d}"
    return (q or d).strip()


def qa_context_text(meta: dict | None, stored_document: str) -> str:
    """Text passed to the model: answer only for QA rows; PDFs unchanged."""
    meta = meta or {}
    if meta.get("kind") != "qa":
        return stored_document
    ans = (meta.get("answer") or "").strip()
    if ans:
        return ans
    doc = stored_document or ""
    if "ምላሽ፦" in doc:
        try:
            return doc.split("ምላሽ፦", 1)[1].strip()
        except IndexError:
            pass
    if "\n\n" in doc:
        parts = doc.split("\n\n", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return doc.strip()
