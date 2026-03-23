from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source_id: str
    chunk_index: int


def split_paragraphs(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source_id: str,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> List[Chunk]:
    """Character-based sliding window. Good enough for course demos; swap for token-based if needed."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    step = max(1, max_chars - overlap_chars)

    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source_id=source_id, chunk_index=idx))
            idx += 1
        if end >= len(text):
            break
        start += step

    return chunks


def chunk_document(full_text: str, source_id: str, max_chars: int, overlap_chars: int) -> List[Chunk]:
    """Prefer paragraph boundaries, then window within merged text."""
    paras = split_paragraphs(full_text)
    if not paras:
        return []
    merged = "\n\n".join(paras)
    return chunk_text(merged, source_id, max_chars=max_chars, overlap_chars=overlap_chars)
