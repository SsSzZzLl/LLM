"""
Retrieval Agent — document retrieval for the Multi-Agent RAG pipeline.

Wraps `rag_qa.retrieve.retrieve` with `AgentInput` / `AgentOutput`, supports
single-query retrieval and multi-query merge when `sub_questions` are present.

Author: 薛玉珩 (Retrieval Agent) — collaborates with `retrieve.py`
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rag_qa.agents.base_agent import AgentInput, AgentOutput, BaseAgent
from rag_qa.index_store import ChunkRecord, DocumentIndex
from rag_qa.retrieve import RetrievalMode, retrieve


class RetrievalAgent(BaseAgent):
    """
    Retrieves passages from a `DocumentIndex` using dense / BM25 / hybrid search.

    `run` returns:
      - `evidence`: list of passage texts (same order as scores).
      - `metadata["passages"]`: ``(stable_id, text)`` pairs for the orchestrator.
      - `metadata["scores"]`: retrieval scores (min-max normalized per query in `retrieve`).
    """

    def __init__(
        self,
        index: DocumentIndex,
        top_k: int = 5,
        mode: RetrievalMode = "dense",
        hybrid_bm25_weight: float = 0.35,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.mode: RetrievalMode = mode
        self.hybrid_bm25_weight = hybrid_bm25_weight

    @classmethod
    def from_config(cls, index: DocumentIndex, cfg: Dict[str, Any]) -> "RetrievalAgent":
        r = cfg.get("retrieval", {}) or {}
        mode = r.get("mode", "dense")
        if mode not in ("dense", "bm25", "hybrid"):
            mode = "dense"
        return cls(
            index=index,
            top_k=int(r.get("top_k", 5)),
            mode=mode,  # type: ignore[arg-type]
            hybrid_bm25_weight=float(r.get("hybrid_bm25_weight", 0.35)),
        )

    def run(self, input_data: AgentInput) -> AgentOutput:
        meta = input_data.metadata or {}
        base_k = int(meta.get("top_k", self.top_k))
        mode = meta.get("mode", self.mode)
        if mode not in ("dense", "bm25", "hybrid"):
            mode = self.mode
        hybrid_w = float(meta.get("hybrid_bm25_weight", self.hybrid_bm25_weight))

        complexity = str(meta.get("complexity", "")).lower()
        use_multi_hop = bool(meta.get("use_multi_hop", False))
        n = len(self.index.records)
        effective_k = base_k
        if complexity == "complex" or use_multi_hop:
            effective_k = min(max(base_k * 2, base_k), n) if n else base_k

        queries = self._queries(input_data)
        # Each query retrieves up to effective_k; results are merged and re-ranked globally.
        per_q_k = effective_k if len(queries) <= 1 else min(effective_k, max(4, effective_k * 2 // len(queries)))

        hits_list: List[List[Tuple[ChunkRecord, float]]] = []
        for q in queries:
            hits_list.append(
                retrieve(
                    self.index,
                    q,
                    top_k=per_q_k,
                    mode=mode,  # type: ignore[arg-type]
                    hybrid_bm25_weight=hybrid_w,
                )
            )

        merged = self._merge_hits(hits_list, top_k=effective_k)

        passages: List[Tuple[str, str]] = [(r.stable_id, r.text) for r, _ in merged]
        scores = [float(s) for _, s in merged]
        evidence = [r.text for r, _ in merged]

        conf = sum(scores) / len(scores) if scores else 0.0

        return AgentOutput(
            answer="",
            evidence=evidence,
            confidence=min(1.0, max(0.0, conf)),
            metadata={
                "passages": passages,
                "scores": scores,
                "mode": mode,
                "queries_used": queries,
                "per_query_top_k": per_q_k,
            },
        )

    def _queries(self, input_data: AgentInput) -> List[str]:
        main = input_data.question.strip()
        subs = [s.strip() for s in input_data.sub_questions if s.strip()]
        if not subs:
            return [main] if main else [""]

        ordered: List[str] = []
        seen: set[str] = set()
        for s in subs:
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
        if main and main not in seen:
            ordered.append(main)
        return ordered or ([main] if main else [""])

    @staticmethod
    def _merge_hits(
        hits_list: List[List[Tuple[ChunkRecord, float]]],
        top_k: int,
    ) -> List[Tuple[ChunkRecord, float]]:
        """Union chunks across queries, keep max score per stable_id, then take top_k."""
        best: Dict[str, Tuple[ChunkRecord, float]] = {}
        for hits in hits_list:
            for rec, score in hits:
                sid = rec.stable_id
                if sid not in best or score > best[sid][1]:
                    best[sid] = (rec, float(score))
        merged = sorted(best.values(), key=lambda x: -x[1])
        return merged[:top_k] if top_k > 0 else []
