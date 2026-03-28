"""
Base Agent definitions for Multi-Agent RAG System.

This module defines the common interfaces and data structures that all
agents in the system must implement and use for communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentInput:
    """Standardized input format for all agents."""
    question: str
    context: List[str] = field(default_factory=list)      # Retrieved passages or previous context
    sub_questions: List[str] = field(default_factory=list) # Decomposed sub-questions
    metadata: Dict[str, Any] = field(default_factory=dict) # Routing config, history, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "context": self.context,
            "sub_questions": self.sub_questions,
            "metadata": self.metadata,
        }


@dataclass
class AgentOutput:
    """Standardized output format for all agents."""
    answer: str
    evidence: List[str] = field(default_factory=list)      # Passages cited/used
    confidence: float = 1.0                                # Confidence in the output (0.0 to 1.0)
    should_retry: bool = False                             # For Self-Reflection: flag to retry
    metadata: Dict[str, Any] = field(default_factory=dict) # E.g., token usage, reasoning steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "should_retry": self.should_retry,
            "metadata": self.metadata,
        }


class BaseAgent:
    """
    Abstract base class that all specialized agents must inherit from.
    """
    
    def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the agent's core block of logic.
        
        Args:
            input_data: The standardized AgentInput payload.
            
        Returns:
            The standardized AgentOutput payload.
        """
        raise NotImplementedError("Each agent must implement the 'run' method.")
