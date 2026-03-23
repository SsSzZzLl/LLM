from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_qa.agents.route_agent import RouteAgent, RouteDecision, QuestionComplexity
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
    route_decision: Optional[RouteDecision] = field(default=None)  # Track routing info


class RAGPipeline:
    def __init__(self, index: DocumentIndex, cfg: Dict[str, Any], route_agent: Optional[RouteAgent] = None) -> None:
        self.index = index
        self.cfg = cfg
        self.route_agent = route_agent

    @classmethod
    def from_disk(cls, cfg: Optional[Dict[str, Any]] = None, use_routing: bool = False) -> "RAGPipeline":
        cfg = cfg or load_config()
        index_dir = resolve_path(cfg, "index_dir")
        index = DocumentIndex.load(index_dir)
        
        # Initialize RouteAgent if dynamic routing is enabled
        route_agent = None
        if use_routing:
            g_cfg = cfg.get("generation", {})
            route_agent = RouteAgent(
                model=g_cfg.get("openai_model"),
                temperature=0.1,
                max_tokens=512,
            )
        
        return cls(index=index, cfg=cfg, route_agent=route_agent)

    def answer_question(
        self, 
        question: str, 
        use_retrieval: bool = True,
        use_dynamic_routing: bool = False,
    ) -> RAGAnswer:
        """
        Answer a question with optional dynamic routing.
        
        Args:
            question: The question to answer
            use_retrieval: Whether to use retrieval (legacy, overridden by routing)
            use_dynamic_routing: Whether to use RouteAgent for dynamic routing
        """
        r_cfg = self.cfg.get("retrieval", {})
        g_cfg = self.cfg.get("generation", {})
        top_k = int(r_cfg.get("top_k", 5))
        mode = r_cfg.get("mode", "dense")
        
        route_decision = None
        
        # Dynamic routing: use RouteAgent to determine strategy
        if use_dynamic_routing and self.route_agent:
            route_decision, routing_config = self.route_agent.route(question)
            
            # Override use_retrieval based on routing decision
            use_retrieval = routing_config["use_retrieval"]
            
            # Log routing info (could be extended to use multi-hop)
            # For now, we use the routing decision but keep single-hop retrieval
            # Multi-hop would be implemented in a more advanced version
            
        elif not use_retrieval:
            # Legacy direct generation path
            ans = generate_chat(
                no_context_system_prompt(),
                no_context_user_prompt(question),
                model=g_cfg.get("openai_model"),
                temperature=float(g_cfg.get("temperature", 0.2)),
                max_tokens=int(g_cfg.get("max_tokens", 512)),
            )
            return RAGAnswer(answer=ans, passages=[], scores=[], route_decision=route_decision)

        # Retrieval-based answering
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
        return RAGAnswer(answer=ans, passages=passages, scores=scores, route_decision=route_decision)


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
