"""Chunk Amharic and mixed text for RAG (sentence-ish boundaries, size caps)."""

from __future__ import annotations

import re
from typing import Iterator

# Amharic full stop and common Latin sentence ends
_SPLIT_RE = re.compile(r"(?<=[።!?])\s+|\n{2,}")


def chunk_text(
    text: str,
    max_chars: int = 1600,
    overlap: int = 200,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    if not parts:
        parts = [text]

    chunks: list[str] = []
    buf = ""

    for p in parts:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = f"{buf} {p}".strip() if buf else p
            continue
        if buf:
            chunks.extend(_roll_window(buf, max_chars, overlap))
        if len(p) <= max_chars:
            buf = p
        else:
            chunks.extend(_hard_split(p, max_chars, overlap))
            buf = ""

    if buf:
        chunks.extend(_roll_window(buf, max_chars, overlap))

    out: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        c = c.strip()
        if len(c) < 30:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _roll_window(buf: str, max_chars: int, overlap: int) -> list[str]:
    if len(buf) <= max_chars:
        return [buf]
    return _hard_split(buf, max_chars, overlap)


def _hard_split(buf: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(buf)
    while start < n:
        end = min(start + max_chars, n)
        piece = buf[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def iter_pdf_pages_text(page_texts: list[str]) -> Iterator[str]:
    for t in page_texts:
        t = (t or "").strip()
        if len(t) < 20:
            continue
        yield t
