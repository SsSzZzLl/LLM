from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class ChunkRecord:
    text: str
    source_id: str
    chunk_index: int

    @property
    def stable_id(self) -> str:
        return f"{self.source_id}#{self.chunk_index}"


def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", s.lower())


class DocumentIndex:
    """Holds chunks, BM25, and dense embeddings. Dense matrix shape (n, dim)."""

    def __init__(
        self,
        records: List[ChunkRecord],
        bm25: BM25Okapi,
        dense: np.ndarray | None,
        embedding_model_name: str,
    ) -> None:
        self.records = records
        self.bm25 = bm25
        self.dense = dense
        self.embedding_model_name = embedding_model_name

    @classmethod
    def build(
        cls,
        records: List[ChunkRecord],
        embedding_model_name: str,
        encode_batch_size: int = 64,
    ) -> "DocumentIndex":
        tokenized = [tokenize(r.text) for r in records]
        bm25 = BM25Okapi(tokenized)

        dense = None
        if embedding_model_name:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_model_name)
            texts = [r.text for r in records]
            dense = model.encode(
                texts,
                batch_size=encode_batch_size,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return cls(records=records, bm25=bm25, dense=dense, embedding_model_name=embedding_model_name)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "embedding_model_name": self.embedding_model_name,
            "n_chunks": len(self.records),
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with open(index_dir / "records.pkl", "wb") as f:
            pickle.dump(self.records, f)
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        if self.dense is not None:
            np.save(index_dir / "dense.npy", self.dense)

    @classmethod
    def load(cls, index_dir: Path) -> "DocumentIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        with open(index_dir / "records.pkl", "rb") as f:
            records: List[ChunkRecord] = pickle.load(f)
        with open(index_dir / "bm25.pkl", "rb") as f:
            bm25: BM25Okapi = pickle.load(f)
        dense_path = index_dir / "dense.npy"
        dense = np.load(dense_path) if dense_path.exists() else None
        return cls(
            records=records,
            bm25=bm25,
            dense=dense,
            embedding_model_name=meta.get("embedding_model_name", ""),
        )


def records_from_chunks(chunks) -> List[ChunkRecord]:
    from rag_qa.chunking import Chunk

    out: List[ChunkRecord] = []
    for c in chunks:
        if isinstance(c, Chunk):
            out.append(ChunkRecord(text=c.text, source_id=c.source_id, chunk_index=c.chunk_index))
        else:
            raise TypeError("Expected Chunk list")
    return out
