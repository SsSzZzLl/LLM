from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np

from rag_qa.index_store import ChunkRecord, DocumentIndex, tokenize

RetrievalMode = Literal["dense", "bm25", "hybrid"]

_MODEL_CACHE: Dict[str, object] = {}


def _dense_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalized rows."""
    return matrix @ query_vec


def _get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        # 优先使用本地缓存，避免每次联网检查
        _MODEL_CACHE[model_name] = SentenceTransformer(
            model_name,
            local_files_only=True,
        )
    return _MODEL_CACHE[model_name]


def retrieve(
    index: DocumentIndex,
    query: str,
    top_k: int,
    mode: RetrievalMode,
    hybrid_bm25_weight: float = 0.35, # This parameter will be ignored for RRF, but kept for compatibility
    rrf_k: int = 60, # Add rrf_k parameter
) -> List[Tuple[ChunkRecord, float]]:
    if top_k <= 0:
        return []

    n = len(index.records)
    if n == 0:
        return []

    q_tokens = tokenize(query)
    bm25_scores = np.array(index.bm25.get_scores(q_tokens), dtype=np.float64)

    dense_scores = None
    if mode in ("dense", "hybrid") and index.dense is not None:
        model = _get_embedding_model(index.embedding_model_name)
        qv = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        dense_scores = _dense_scores(qv, index.dense)

    if mode == "bm25":
        combined_scores = bm25_scores
        order = np.argsort(-combined_scores)[:top_k]
    elif mode == "dense":
        if dense_scores is None:
            raise ValueError("Dense retrieval requested but index has no dense vectors.")
        combined_scores = dense_scores
        order = np.argsort(-combined_scores)[:top_k]
    else: # mode == "hybrid"
        if dense_scores is None:
            raise ValueError("Hybrid requires both BM25 and dense index.")
        
        bm25_ranks = np.argsort(-bm25_scores).tolist()
        dense_ranks = np.argsort(-dense_scores).tolist()

        fused_indices = rrf([bm25_ranks, dense_ranks], k=rrf_k)
        
        order = fused_indices[:top_k]
        
        combined_scores = np.zeros(n)
        for rank, doc_idx in enumerate(order):
            combined_scores[doc_idx] = 1.0 - (rank / len(order)) 

    return [(index.records[int(i)], float(combined_scores[int(i)])) for i in order]


def rrf(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """
    Reciprocal Rank Fusion (RRF) algorithm.
    rank_lists: A list of lists, where each inner list contains document indices ordered by rank.
    k: A constant to control the contribution of individual ranks.
    """
    fused_scores = {}
    for rank_list in rank_lists:
        for rank, doc_idx in enumerate(rank_list):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0) + 1 / (k + rank + 1) # rank is 0-indexed, so add 1

    # Sort documents by their fused scores in descending order
    sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_idx for doc_idx, _ in sorted_docs]


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)