from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_qa.config import load_config, resolve_path
from rag_qa.generate import generate_chat
from rag_qa.index_store import DocumentIndex, records_from_chunks
from rag_qa.ingest import load_corpus
from rag_qa.prompts import (
    format_context_passages,
    no_context_system_prompt,
    no_context_user_prompt,
    rag_system_prompt,
    rag_user_prompt,
)
from rag_qa.retrieve import retrieve


@dataclass
class RAGAnswer:
    answer: str
    passages: List[Tuple[str, str]]  # (stable_id, text)
    scores: List[float]


class RAGPipeline:
    def __init__(self, index: DocumentIndex, cfg: Dict[str, Any]) -> None:
        self.index = index
        self.cfg = cfg

    @classmethod
    def from_disk(cls, cfg: Optional[Dict[str, Any]] = None) -> "RAGPipeline":
        cfg = cfg or load_config()
        index_dir = resolve_path(cfg, "index_dir")
        index = DocumentIndex.load(index_dir)
        return cls(index=index, cfg=cfg)

    def answer_question(self, question: str, use_retrieval: bool = True) -> RAGAnswer:
        r_cfg = self.cfg.get("retrieval", {})
        g_cfg = self.cfg.get("generation", {})
        top_k = int(r_cfg.get("top_k", 5))
        mode = r_cfg.get("mode", "dense")

        if not use_retrieval:
            ans = generate_chat(
                no_context_system_prompt(),
                no_context_user_prompt(question),
                model=g_cfg.get("openai_model"),
                temperature=float(g_cfg.get("temperature", 0.2)),
                max_tokens=int(g_cfg.get("max_tokens", 512)),
            )
            return RAGAnswer(answer=ans, passages=[], scores=[])

        hybrid_w = float(r_cfg.get("hybrid_bm25_weight", 0.35))
        hits = retrieve(
            self.index,
            question,
            top_k=top_k,
            mode=mode,
            hybrid_bm25_weight=hybrid_w,
        )
        passages = [(h[0].stable_id, h[0].text) for h in hits]
        scores = [h[1] for h in hits]
        ctx = format_context_passages(passages)
        ans = generate_chat(
            rag_system_prompt(),
            rag_user_prompt(question, ctx),
            model=g_cfg.get("openai_model"),
            temperature=float(g_cfg.get("temperature", 0.2)),
            max_tokens=int(g_cfg.get("max_tokens", 512)),
        )
        return RAGAnswer(answer=ans, passages=passages, scores=scores)


def build_index_from_corpus(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    corpus_dir = resolve_path(cfg, "corpus_dir")
    index_dir = resolve_path(cfg, "index_dir")
    ch = cfg.get("chunking", {})
    emb = cfg.get("embedding", {})

    chunks = load_corpus(
        corpus_dir,
        max_chars=int(ch.get("max_chars", 900)),
        overlap_chars=int(ch.get("overlap_chars", 120)),
    )
    records = records_from_chunks(chunks)
    index = DocumentIndex.build(
        records,
        embedding_model_name=str(emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
    )
    index.save(index_dir)
    return index_dir
