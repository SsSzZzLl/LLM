from __future__ import annotations

from pathlib import Path
from typing import List

from rag_qa.chunking import Chunk, chunk_document


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_corpus(corpus_dir: Path, max_chars: int, overlap_chars: int) -> List[Chunk]:
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    all_chunks: List[Chunk] = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            text = read_text_file(p)
            source_id = str(p.relative_to(corpus_dir)).replace("\\", "/")
            all_chunks.extend(chunk_document(text, source_id, max_chars, overlap_chars))
    return all_chunks
