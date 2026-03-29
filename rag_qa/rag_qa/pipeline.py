from __future__ import annotations

import logging
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
# Import new Agent framework components
from rag_qa.agents import AgentInput, RetrievalAgent, RouteAgent, QuestionComplexity, ReasoningAgent, SynthesisAgent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    answer: str
    passages: List[Tuple[str, str]]  # (stable_id, text)
    scores: List[float]
    routing_strategy: str
    complexity: str


class MultiAgentOrchestrator:
    """
    Main Orchestrator for the Multi-Agent RAG System.
    Uses the RouteAgent to classify questions and directs them to the 
    appropriate specialized agents (or legacy pipeline logic temporarily).
    """

    def __init__(self, index: DocumentIndex, cfg: Dict[str, Any]) -> None:
        self.index = index
        self.cfg = cfg
        
        # Initialize specialized Agents
        g_cfg = self.cfg.get("generation", {})
        self.route_agent = RouteAgent(
            model=g_cfg.get("openai_model"),
            temperature=float(g_cfg.get("temperature", 0.1)),
        )
        self.retrieval_agent = RetrievalAgent.from_config(self.index, self.cfg)
        
        self.synthesis_agent = SynthesisAgent(
            model=g_cfg.get("openai_model"),
            temperature=float(g_cfg.get("temperature", 0.2)),
            max_tokens=int(g_cfg.get("max_tokens", 512)),
        )
        
        self.reasoning_agent = ReasoningAgent(
            model=g_cfg.get("openai_model"),
            temperature=float(g_cfg.get("temperature", 0.2)),
            max_tokens=int(g_cfg.get("max_tokens", 1024)),
        )

    @classmethod
    def from_disk(cls, cfg: Optional[Dict[str, Any]] = None) -> "MultiAgentOrchestrator":
        cfg = cfg or load_config()
        index_dir = resolve_path(cfg, "index_dir")
        index = DocumentIndex.load(index_dir)
        return cls(index=index, cfg=cfg)

    def answer_question(self, question: str, use_routing: bool = True) -> OrchestratorResponse:
        """
        Answers a user question, dynamically routing it if use_routing=True.
        """
        r_cfg = self.cfg.get("retrieval", {})
        g_cfg = self.cfg.get("generation", {})
        top_k = int(r_cfg.get("top_k", 5))
        mode = r_cfg.get("mode", "dense")
        hybrid_w = float(r_cfg.get("hybrid_bm25_weight", 0.35))

        # 1. Routing phase
        if not use_routing:
            # Fallback to standard RAG if routing is disabled
            decision = None
            complexity = QuestionComplexity.MODERATE
            strategy = "single_hop_rag (routing disabled)"
            route_config = {
                "use_multi_hop": False,
                "strategy_name": strategy,
            }
        else:
            decision, route_config = self.route_agent.route(question)
            complexity = decision.complexity
            strategy = route_config["strategy_name"]

        # 2. Execution phase based on Route Decision
        
        if complexity == QuestionComplexity.SIMPLE:
            # No retrieval needed, answer directly
            ans = generate_chat(
                no_context_system_prompt(),
                no_context_user_prompt(question),
                model=g_cfg.get("openai_model"),
                temperature=float(g_cfg.get("temperature", 0.2)),
                max_tokens=int(g_cfg.get("max_tokens", 512)),
            )
            return OrchestratorResponse(
                answer=ans, passages=[], scores=[], 
                routing_strategy=strategy, complexity=complexity.value
            )
            
        # Retrieval Phase (handles both MODERATE and COMPLEX dynamically)
        is_complex = (complexity == QuestionComplexity.COMPLEX)
        use_multi_hop = bool(route_config.get("use_multi_hop", False))
        
        retrieval_in = AgentInput(
            question=question,
            metadata={
                "top_k": top_k,
                "mode": mode,
                "hybrid_bm25_weight": hybrid_w,
                "complexity": complexity.value if decision else "moderate",
                "use_multi_hop": use_multi_hop,
            },
        )
        retrieval_out = self.retrieval_agent.run(retrieval_in)
        passages = retrieval_out.metadata.get("passages", [])
        scores = retrieval_out.metadata.get("scores", [])
        evidence_texts = retrieval_out.evidence

        # Generation/Synthesis Phase
        if is_complex or use_multi_hop:
            agent_in = AgentInput(
                question=question,
                context=evidence_texts,
                metadata=route_config,
            )
            agent_out = self.reasoning_agent.run(agent_in)
            final_answer = agent_out.answer
        else:
            agent_in = AgentInput(
                question=question,
                context=evidence_texts,
                metadata=route_config,
            )
            agent_out = self.synthesis_agent.run(agent_in)
            final_answer = agent_out.answer
            
        return OrchestratorResponse(
            answer=final_answer, 
            passages=passages, 
            scores=scores,
            routing_strategy=strategy, 
            complexity=complexity.value if decision else "moderate"
        )


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
