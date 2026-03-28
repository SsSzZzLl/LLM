from __future__ import annotations

import re
import tiktoken
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
    max_tokens: int = 256,
    overlap_tokens: int = 30,
    model_name: str = "cl100k_base"
) -> List[Chunk]:
    """Token-based sliding window chunking."""
    if not text:
        return []

    encoding = tiktoken.get_encoding(model_name)
    tokens = encoding.encode(text)

    chunks: List[Chunk] = []
    start_token_idx = 0
    chunk_idx = 0
    
    step = max(1, max_tokens - overlap_tokens)

    while start_token_idx < len(tokens):
        end_token_idx = min(start_token_idx + max_tokens, len(tokens))
        
        chunk_tokens = tokens[start_token_idx:end_token_idx]
        chunk_text_content = encoding.decode(chunk_tokens)
        
        if chunk_text_content.strip():
            chunks.append(Chunk(text=chunk_text_content, source_id=source_id, chunk_index=chunk_idx))
            chunk_idx += 1
        
        if end_token_idx >= len(tokens):
            break
        
        start_token_idx += step

    return chunks


def chunk_document(full_text: str, source_id: str, max_tokens: int, overlap_tokens: int, model_name: str = "cl100k_base") -> List[Chunk]:
    """Prefer paragraph boundaries, then window within merged text, using token-based chunking."""
    paras = split_paragraphs(full_text)
    if not paras:
        return []
    merged = "\n\n".join(paras)
    return chunk_text(merged, source_id, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model_name=model_name)
