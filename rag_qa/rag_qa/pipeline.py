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
from rag_qa.agents.critic_agent import CriticAgent

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    answer: str
    passages: List[Tuple[str, str]]  # (stable_id, text)
    scores: List[float]
    routing_strategy: str
    complexity: str
    trace: str = "" # To expose the inner thoughts
    telemetry: dict = None


class MultiAgentOrchestrator:
    """
    Main Orchestrator for the Multi-Agent RAG System.
    Uses the RouteAgent to classify questions and directs them to the 
    appropriate specialized agents.
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
            max_tokens=int(g_cfg.get("max_tokens", 2048)),
        )
        
        self.reasoning_agent = ReasoningAgent(
            model=g_cfg.get("openai_model"),
            temperature=float(g_cfg.get("temperature", 0.2)),
            max_tokens=int(g_cfg.get("max_tokens", 2048)),
        )
        
        # New Critic Agent
        self.critic_agent = CriticAgent(
            model=g_cfg.get("openai_model"),
            temperature=0.1,
            max_tokens=int(g_cfg.get("max_tokens", 2048)),
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
        
        telemetry = {"total_latency": 0.0, "total_prompt_tokens": 0, "total_completion_tokens": 0}
        def _add_tel(t):
            if t:
                telemetry["total_latency"] += t.get("total_latency", t.get("latency", 0))
                telemetry["total_prompt_tokens"] += t.get("total_prompt_tokens", t.get("prompt_tokens", 0))
                telemetry["total_completion_tokens"] += t.get("total_completion_tokens", t.get("completion_tokens", 0))

        # 1. Routing phase
        if not use_routing:
            decision = None
            complexity = QuestionComplexity.MODERATE
            strategy = "single_hop_rag (routing disabled)"
            route_config = {"use_multi_hop": False, "strategy_name": strategy}
        else:
            decision, route_config = self.route_agent.route(question)
            complexity = decision.complexity
            strategy = route_config["strategy_name"]
            _add_tel(decision.telemetry)

        # 2. Execution phase based on Route Decision
        
        if complexity == QuestionComplexity.SIMPLE:
            ans = generate_chat(
                no_context_system_prompt(),
                no_context_user_prompt(question),
                model=g_cfg.get("openai_model"),
                temperature=float(g_cfg.get("temperature", 0.2)),
                max_tokens=int(g_cfg.get("max_tokens", 2048)),
            )
            try:
                # Try parsing JSON if simple mode used it
                import json
                ans_dict = json.loads(re.search(r'\{.*\}', ans, re.DOTALL).group())
                exact_ans = ans_dict.get("exact_answer", ans)
                trace_str = ans_dict.get("thought_process", "")
            except:
                exact_ans = ans
                trace_str = ""
            return OrchestratorResponse(
                answer=exact_ans, passages=[], scores=[], 
                routing_strategy=strategy, complexity=complexity.value,
                trace=trace_str, telemetry=telemetry
            )
            
        is_complex = (complexity == QuestionComplexity.COMPLEX)
        use_multi_hop = bool(route_config.get("use_multi_hop", False))
        
        scratchpad_trace = ""
        passages = []
        scores = []
        
        if is_complex or use_multi_hop:
            # ReAct Loop for Complex Questions!
            # Attach retriever lambda
            def _fetcher(q, top_k=3):
                ret_out = self.retrieval_agent.run(AgentInput(
                    question=q, 
                    metadata={"top_k": top_k, "mode": mode, "hybrid_bm25_weight": hybrid_w}
                ))
                return ret_out.evidence
                
            self.reasoning_agent.set_retriever(_fetcher)
            
            critic_feedback = ""
            max_loops = 2
            
            for loop_i in range(max_loops):
                # 1. Planner & Worker Phase
                agent_in = AgentInput(
                    question=question,
                    metadata={
                        "scratchpad_trace": scratchpad_trace,
                        "critic_feedback": critic_feedback
                    }
                )
                reason_out = self.reasoning_agent.run(agent_in)
                scratchpad_trace = reason_out.metadata["full_trace"]
                _add_tel(reason_out.metadata.get("telemetry", {}))
                
                # 2. Critic Phase
                critic_in = AgentInput(
                    question=question, 
                    metadata={"scratchpad_trace": scratchpad_trace}
                )
                critic_out = self.critic_agent.run(critic_in)
                critic_feedback = critic_out.metadata["critic_feedback"]
                _add_tel(critic_out.metadata.get("telemetry", {}))
                
                scratchpad_trace += f"\n\n[Critic Evaluation Loop {loop_i+1}]: {critic_feedback}\n\n"
                
                if not critic_out.should_retry:
                    # Critic approved!
                    break
                    
            final_context = [scratchpad_trace]
            
        else:
            # Moderate - Single Hop RAG Pipeline
            retrieval_in = AgentInput(
                question=question,
                metadata={
                    "top_k": top_k, "mode": mode, "hybrid_bm25_weight": hybrid_w,
                    "complexity": complexity.value, "use_multi_hop": False
                },
            )
            retrieval_out = self.retrieval_agent.run(retrieval_in)
            passages = retrieval_out.metadata.get("passages", [])
            scores = retrieval_out.metadata.get("scores", [])
            final_context = retrieval_out.evidence
            scratchpad_trace = "Single-hop retrieved. No reasoning trace.\n"

        # Synthesis Phase
        agent_in = AgentInput(
            question=question,
            context=final_context,
            metadata=route_config,
        )
        agent_out = self.synthesis_agent.run(agent_in)
        raw_final_json_str = agent_out.answer 
        _add_tel(agent_out.metadata.get("telemetry", {})) 
        
        # Safely parse exact answer
        import json
        try:
            parsed = json.loads(raw_final_json_str)
            final_answer = parsed.get("exact_answer", raw_final_json_str)
            final_thoughts = parsed.get("thought_process", "")
        except:
            final_answer = raw_final_json_str
            final_thoughts = ""
            
        combined_trace = scratchpad_trace + f"\n\n[Final Synthesis Thoughts]:\n{final_thoughts}"

        return OrchestratorResponse(
            answer=final_answer, 
            passages=passages, 
            scores=scores,
            routing_strategy=strategy, 
            complexity=complexity.value if decision else "moderate",
            trace=combined_trace,
            telemetry=telemetry,
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
