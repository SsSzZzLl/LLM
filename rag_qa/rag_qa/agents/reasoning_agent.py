"""
Reasoning Agent - Multi-hop Reasoning and Self-Reflection for RAG System

This module implements the ReasoningAgent which:
1. Decomposes complex multi-hop questions into sub-questions
2. Implements Self-Reflection mechanism to evaluate answer quality
3. Decides whether to retry with modified strategy

Author: 叶子冉 (Reasoning Agent Developer)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_qa.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from rag_qa.generate import generate_chat


@dataclass
class SubQuestion:
    """Represents a decomposed sub-question with its dependencies."""
    id: str
    question: str
    dependencies: List[str] = field(default_factory=list)
    answer: str = ""
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "dependencies": self.dependencies,
            "answer": self.answer,
            "evidence": self.evidence,
        }


@dataclass
class DecompositionResult:
    """Result of question decomposition."""
    sub_questions: List[SubQuestion]
    reasoning_plan: str
    estimated_hops: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "reasoning_plan": self.reasoning_plan,
            "estimated_hops": self.estimated_hops,
        }


@dataclass
class ReflectionResult:
    """Result of self-reflection on answer quality."""
    is_satisfactory: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    should_retry: bool
    retry_strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_satisfactory": self.is_satisfactory,
            "confidence": self.confidence,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "should_retry": self.should_retry,
            "retry_strategy": self.retry_strategy,
        }


class ReasoningAgent(BaseAgent):
    """
    Reasoning Agent for multi-hop question decomposition acting as Planner and Worker.
    
    This agent:
    1. Breaks down complex questions into sub-questions (Plan)
    2. Uses the provided retrieval mechanism to gather facts for each sub-question (Work)
    3. Appends all findings to a shared Scratchpad Trace for the Critic to evaluate
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Inject retrieval function dynamically from Orchestrator
        self.retrieving_func = None
    
    def set_retriever(self, func):
        self.retrieving_func = func

    def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute Planner and Worker phases and return the generated Scratchpad Trace.
        """
        question = input_data.question
        
        # Any prior trace that Critic rejected
        prior_trace = input_data.metadata.get("scratchpad_trace", "")
        critic_feedback = input_data.metadata.get("critic_feedback", "")
        
        # Telemetry aggregation
        telemetry = {"total_latency": 0.0, "total_prompt_tokens": 0, "total_completion_tokens": 0}
        def _add_tel(t: dict):
            if t:
                telemetry["total_latency"] += t.get("latency", 0)
                telemetry["total_prompt_tokens"] += t.get("prompt_tokens", 0)
                telemetry["total_completion_tokens"] += t.get("completion_tokens", 0)

        # Planner Phase
        planner_tracker = {"agent_name": "Planner"}
        plan = self._generate_plan(question, prior_trace, critic_feedback, planner_tracker)
        _add_tel(planner_tracker)
        
        # Worker Phase
        new_trace = f"--- Worker Execution Plan ---\n{plan}\n[⏱️ Planner Latency: {planner_tracker.get('latency', 0):.2f}s | 🪙 Tokens: {planner_tracker.get('prompt_tokens',0)} in, {planner_tracker.get('completion_tokens',0)} out]\n\n"
        
        # Extract searches to perform based on plan (heuristic simple parse)
        steps_prompt = self._get_worker_prompt()
        user_prompt = f"Original Question: {question}\nPlan: {plan}\nPrior Trace: {prior_trace}\nCritic Feedback: {critic_feedback}\n\nExecute the plan step-by-step and write out your findings."
        worker_tracker = {"agent_name": "Worker"}
        try:
            worker_response = generate_chat(
                system=steps_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tracker=worker_tracker,
            )
            _add_tel(worker_tracker)
            
            sub_queries = re.findall(r"SEARCH:\s*(.*)", worker_response)
            if not sub_queries and not prior_trace:
                sub_queries = [question] # Fallback
                
            evidence_gathered = []
            if self.retrieving_func:
                for sq in sub_queries:
                    chunks = self.retrieving_func(sq, top_k=3)
                    evidence_gathered.extend(chunks)
                    
            if evidence_gathered:
                new_trace += "\n--- Retrieved Context ---\n"
                for i, ev in enumerate(evidence_gathered[:5]):
                    new_trace += f"[{i+1}] {ev}\n"
                    
            new_trace += f"\n--- Worker Thoughts ---\n{worker_response}\n[⏱️ Worker Latency: {worker_tracker.get('latency', 0):.2f}s | 🪙 Tokens: {worker_tracker.get('prompt_tokens',0)} in, {worker_tracker.get('completion_tokens',0)} out]\n"
            
        except Exception as e:
            new_trace += f"\nFailed execution: {str(e)}"
            
        full_trace = prior_trace + new_trace
            
        return AgentOutput(
            answer="Planner/Worker execution complete. See trace.",
            evidence=[],
            confidence=1.0,
            should_retry=False,
            metadata={
                "new_evidence": new_trace,
                "full_trace": full_trace,
                "telemetry": telemetry
            }
        )

    def _generate_plan(self, question: str, prior_trace: str, critic_feedback: str, tracker: dict) -> str:
        system = "You are a Planner Agent. Decompose the complex question into 2-3 step-by-step search operations."
        user = f"Question: {question}\n\nPrior Trace: {prior_trace}\nCritic Feedback: {critic_feedback}\n\nWrite a numbered list of sub-questions to search for. Start each search target with 'SEARCH: ' (e.g., SEARCH: When was X born?)"
        
        try:
            return generate_chat(system, user, self.model, 0.2, 256, tracker=tracker)
        except:
            return f"1. SEARCH: {question}"

    def _get_worker_prompt(self) -> str:
        return "You are a Worker Agent. Follow the Plan and execute reasoning. You may use SEARCH requests for missing facts."

