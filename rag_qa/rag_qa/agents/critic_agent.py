"""
Critic Agent - Multi-Agent RAG System

This module implements the CriticAgent which acts as the 'Judge' in the ReAct loop,
evaluating whether the current accumulated evidence is sufficient to directly 
answer the original complex question.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from rag_qa.agents.base_agent import AgentInput, AgentOutput, BaseAgent
from rag_qa.generate import generate_chat


class CriticAgent(BaseAgent):
    """
    Critic Agent for evaluating evidence sufficiency and logic consistency.
    
    If evidence is missing or there's a break in the multi-hop logic,
    the Critic returns a failure and provides suggestions for the Planner.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Evaluate the accumulated trace to decide if synthesizing an answer is safe.
        
        Args:
            input_data: AgentInput containing the original question and the 
                        accumulated evidence/thoughts in `metadata["scratchpad_trace"]`.
                        
        Returns:
            AgentOutput where `should_retry` determines if Planner needs to run again.
            `metadata["critic_feedback"]` contains the specific suggestions.
        """
        question = input_data.question
        trace = input_data.metadata.get("scratchpad_trace", "")
        
        if not trace:
            return AgentOutput(
                answer="No trace provided.",
                evidence=[],
                confidence=0.0,
                should_retry=True,
                metadata={"critic_feedback": "Initial step, no evidence gathered yet."}
            )

        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(question, trace)
        
        tracker = {"agent_name": "Critic"}

        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tracker=tracker,
            )
            
            is_sufficient, feedback = self._parse_response(response)
            
        except Exception as e:
            # Fallback
            is_sufficient = False
            feedback = f"Critic parsing failed: {str(e)}"

        return AgentOutput(
            answer="Critic evaluation complete.",
            evidence=[],
            confidence=1.0 if is_sufficient else 0.0,
            should_retry=not is_sufficient,
            metadata={
                "critic_feedback": feedback,
                "telemetry": tracker,
            }
        )

    def _get_system_prompt(self) -> str:
        return """You are the Critic (Judge) in a Multi-Agent system attempting to answer complex multi-hop questions.
Your job is to read the Original Question and the Accumulated Evidence Trace gathered so far by the Worker agents.

You must answer ONE central question: "Do we have ALL the necessary exact facts to decisively answer the Original Question?"

For example:
- If Q is "When was Trump's wife born?" and the trace only proves "Trump's wife is Melania", you MUST return {"is_sufficient": false} because Melania's birth date is still completely missing.
- If the trace has both facts, return {"is_sufficient": true}.

IMPORTANT: You must output strictly valid JSON in the following format:
```json
{
  "is_sufficient": true/false,
  "feedback": "<Explain why it is sufficient, or precisely what entity/fact is missing and needs to be searched next>"
}
```
Do NOT wrap the JSON in markdown code blocks. Just output raw JSON.
"""

    def _build_user_prompt(self, question: str, trace: str) -> str:
        return (
            f"Original Question to Answer: {question}\n\n"
            "--- Accumulated Evidence Trace ---\n"
            f"{trace}\n"
            "----------------------------------\n\n"
            "Analyze the trace carefully. Does the trace contain the final exact answer to the Original Question? Output JSON only."
        )

    def _parse_response(self, response: str) -> tuple[bool, str]:
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
                
            is_sufficient = data.get("is_sufficient", False)
            feedback = data.get("feedback", "No feedback provided.")
            return is_sufficient, feedback
            
        except Exception:
            # If JSON parsing fails, do text heuristic
            response_lower = response.lower()
            is_sufficient = "true" in response_lower and "false" not in response_lower
            return is_sufficient, response.strip()
