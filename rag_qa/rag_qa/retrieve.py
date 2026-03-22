from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np

from rag_qa.index_store import ChunkRecord, DocumentIndex, tokenize

RetrievalMode = Literal["dense", "bm25", "hybrid"]


def _dense_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalized rows."""
    return matrix @ query_vec


def retrieve(
    index: DocumentIndex,
    query: str,
    top_k: int,
    mode: RetrievalMode,
    hybrid_bm25_weight: float = 0.35,
) -> List[Tuple[ChunkRecord, float]]:
    if top_k <= 0:
        return []

    n = len(index.records)
    if n == 0:
        return []

    q_tokens = tokenize(query)
    bm25_scores = np.array(index.bm25.get_scores(q_tokens), dtype=np.float64)
    bm25_norm = _minmax(bm25_scores) if mode in ("bm25", "hybrid") else None

    dense_scores = None
    dense_norm = None
    if mode in ("dense", "hybrid") and index.dense is not None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(index.embedding_model_name)
        qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        dense_scores = _dense_scores(qv, index.dense)
        dense_norm = _minmax(dense_scores)

    if mode == "bm25":
        combined = bm25_norm
    elif mode == "dense":
        if dense_norm is None:
            raise ValueError("Dense retrieval requested but index has no dense vectors.")
        combined = dense_norm
    else:
        if dense_norm is None or bm25_norm is None:
            raise ValueError("Hybrid requires both BM25 and dense index.")
        w = hybrid_bm25_weight
        combined = w * bm25_norm + (1.0 - w) * dense_norm

    order = np.argsort(-combined)[:top_k]
    return [(index.records[int(i)], float(combined[int(i)])) for i in order]


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
