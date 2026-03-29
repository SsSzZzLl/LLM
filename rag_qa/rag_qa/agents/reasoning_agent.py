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
    Reasoning Agent for multi-hop question decomposition and self-reflection.
    
    This agent:
    1. Breaks down complex questions into manageable sub-questions
    2. Tracks dependencies between sub-questions
    3. Evaluates answer quality through self-reflection
    4. Decides whether to retry with modified strategy
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 2,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize ReasoningAgent.
        
        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens for generation
            max_retries: Maximum number of retry attempts
            confidence_threshold: Minimum confidence for satisfactory answer
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold
    
    def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the reasoning process.
        
        Args:
            input_data: AgentInput containing question and context
            
        Returns:
            AgentOutput with answer, evidence, and confidence
        """
        question = input_data.question
        context = input_data.context
        
        # Step 1: Decompose question into sub-questions
        decomposition = self.decompose_question(question, context)
        
        # Step 2: Solve sub-questions iteratively
        sub_answers = self._solve_sub_questions(decomposition.sub_questions, context)
        
        # Step 3: Synthesize final answer
        answer = self._synthesize_answer(question, decomposition, sub_answers, context)
        
        # Step 4: Self-reflection on answer quality
        reflection = self.reflect(question, answer, context)
        
        # Collect evidence from all sub-questions
        all_evidence = []
        for sq in decomposition.sub_questions:
            all_evidence.extend(sq.evidence)
        
        return AgentOutput(
            answer=answer,
            evidence=list(set(all_evidence)),
            confidence=reflection.confidence,
            should_retry=reflection.should_retry,
            metadata={
                "decomposition": decomposition.to_dict(),
                "reflection": reflection.to_dict(),
                "sub_answers": sub_answers,
                "retry_count": 0,
            }
        )
    
    def decompose_question(
        self,
        question: str,
        context: List[str],
    ) -> DecompositionResult:
        """
        Decompose a complex question into sub-questions.
        
        Args:
            question: The complex multi-hop question
            context: Retrieved context passages
            
        Returns:
            DecompositionResult with sub-questions and reasoning plan
        """
        context_block = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        system_prompt = self._get_decomposition_prompt()
        user_prompt = f"""Question: {question}

Context Passages:
{context_block}

Please decompose this question into sub-questions following the instructions above."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            decomposition = self._parse_decomposition_response(response)
            return decomposition
            
        except Exception as e:
            # Fallback: return single sub-question
            return DecompositionResult(
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=question,
                        dependencies=[],
                    )
                ],
                reasoning_plan=f"Decomposition failed: {str(e)}. Using original question.",
                estimated_hops=1,
            )
    
    def reflect(
        self,
        question: str,
        answer: str,
        context: List[str],
        sub_questions: Optional[List[SubQuestion]] = None,
    ) -> ReflectionResult:
        """
        Perform self-reflection on answer quality.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Context passages used
            sub_questions: Optional sub-questions and their answers
            
        Returns:
            ReflectionResult with quality assessment and retry decision
        """
        system_prompt = self._get_reflection_prompt()
        
        context_block = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        sub_q_block = ""
        if sub_questions:
            sub_q_block = "\n\nSub-questions and answers:\n"
            for sq in sub_questions:
                sub_q_block += f"- {sq.question}\n  Answer: {sq.answer}\n"
        
        user_prompt = f"""Question: {question}

Generated Answer: {answer}

Context Passages:
{context_block}{sub_q_block}

Please evaluate the answer quality following the instructions above."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=512,
            )
            
            reflection = self._parse_reflection_response(response)
            return reflection
            
        except Exception as e:
            # Fallback: assume satisfactory
            return ReflectionResult(
                is_satisfactory=True,
                confidence=0.5,
                issues=[f"Reflection failed: {str(e)}"],
                suggestions=["Proceed with caution"],
                should_retry=False,
            )
    
    def retry_with_strategy(
        self,
        input_data: AgentInput,
        previous_output: AgentOutput,
        reflection: ReflectionResult,
    ) -> AgentOutput:
        """
        Retry with modified strategy based on reflection feedback.
        
        Args:
            input_data: Original input
            previous_output: Previous agent output
            reflection: Reflection result with suggestions
            
        Returns:
            New AgentOutput with improved answer
        """
        retry_count = previous_output.metadata.get("retry_count", 0)
        
        if retry_count >= self.max_retries:
            # Max retries reached, return previous answer
            return AgentOutput(
                answer=previous_output.answer,
                evidence=previous_output.evidence,
                confidence=previous_output.confidence * 0.9,
                should_retry=False,
                metadata={
                    **previous_output.metadata,
                    "retry_count": retry_count,
                    "max_retries_reached": True,
                }
            )
        
        # Modify strategy based on reflection suggestions
        modified_input = self._modify_input_for_retry(input_data, reflection)
        
        # Retry
        new_output = self.run(modified_input)
        
        # Update retry count
        new_output.metadata["retry_count"] = retry_count + 1
        new_output.metadata["previous_confidence"] = previous_output.confidence
        
        return new_output
    
    def _solve_sub_questions(
        self,
        sub_questions: List[SubQuestion],
        context: List[str],
    ) -> Dict[str, str]:
        """
        Solve sub-questions iteratively, respecting dependencies.
        
        Args:
            sub_questions: List of sub-questions to solve
            context: Context passages
            
        Returns:
            Dictionary mapping sub-question IDs to answers
        """
        answers = {}
        context_block = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        # Sort sub-questions by dependency order (simple topological sort)
        solved = set()
        remaining = set(sq.id for sq in sub_questions)
        
        while remaining:
            progress = False
            for sq in sub_questions:
                if sq.id in remaining:
                    # Check if all dependencies are solved
                    if all(dep in solved for dep in sq.dependencies):
                        # Solve this sub-question
                        answer = self._answer_sub_question(sq, answers, context_block)
                        answers[sq.id] = answer
                        sq.answer = answer
                        solved.add(sq.id)
                        remaining.remove(sq.id)
                        progress = True
            
            if not progress and remaining:
                # Circular dependency or missing dependency, break
                break
        
        return answers
    
    def _answer_sub_question(
        self,
        sub_question: SubQuestion,
        previous_answers: Dict[str, str],
        context_block: str,
    ) -> str:
        """Answer a single sub-question using context and previous answers."""
        system_prompt = (
            "You are answering a sub-question as part of a multi-hop reasoning process. "
            "Answer concisely based on the provided context and any previous sub-question answers."
        )
        
        # Include previous answers if there are dependencies
        dep_block = ""
        if sub_question.dependencies:
            dep_block = "\n\nPrevious answers:\n"
            for dep_id in sub_question.dependencies:
                if dep_id in previous_answers:
                    dep_block += f"- {dep_id}: {previous_answers[dep_id]}\n"
        
        user_prompt = f"""Context:
{context_block}{dep_block}

Sub-question: {sub_question.question}

Provide a concise answer."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=256,
            )
            return response.strip()
        except Exception as e:
            return f"[Error answering sub-question: {str(e)}]"
    
    def _synthesize_answer(
        self,
        original_question: str,
        decomposition: DecompositionResult,
        sub_answers: Dict[str, str],
        context: List[str],
    ) -> str:
        """Synthesize final answer from sub-question answers."""
        system_prompt = (
            "You are synthesizing a final answer from sub-question answers. "
            "Provide a comprehensive answer to the original question based on the sub-answers."
        )
        
        sub_answers_block = "\n".join([
            f"- {sq.question}\n  Answer: {sub_answers.get(sq.id, 'Not answered')}"
            for sq in decomposition.sub_questions
        ])
        
        user_prompt = f"""Original Question: {original_question}

Reasoning Plan: {decomposition.reasoning_plan}

Sub-questions and Answers:
{sub_answers_block}

Synthesize a comprehensive final answer."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.strip()
        except Exception as e:
            # Fallback: concatenate sub-answers
            return " ".join(sub_answers.values())
    
    def _modify_input_for_retry(
        self,
        input_data: AgentInput,
        reflection: ReflectionResult,
    ) -> AgentInput:
        """Modify input based on reflection suggestions for retry."""
        # Add reflection suggestions to metadata
        modified_metadata = {
            **input_data.metadata,
            "reflection_issues": reflection.issues,
            "reflection_suggestions": reflection.suggestions,
            "retry_strategy": reflection.retry_strategy,
        }
        
        return AgentInput(
            question=input_data.question,
            context=input_data.context,
            sub_questions=input_data.sub_questions,
            metadata=modified_metadata,
        )
    
    def _get_decomposition_prompt(self) -> str:
        """Get the system prompt for question decomposition."""
        return """You are a Reasoning Agent in a Multi-Agent RAG system. Your task is to decompose complex multi-hop questions into simpler sub-questions.

## Decomposition Guidelines

1. **Identify the type of reasoning required**:
   - **Comparison**: Questions asking to compare two or more entities
   - **Bridge**: Questions requiring connecting information across multiple passages
   - **Temporal**: Questions involving time-based reasoning
   - **Causal**: Questions asking about causes or effects

2. **Create sub-questions that**:
   - Are simpler and more focused than the original
   - Can be answered independently (where possible)
   - Build toward answering the original question
   - Have clear dependencies if order matters

3. **Assign dependencies**:
   - Use empty list `[]` for independent questions
   - List IDs of sub-questions that must be answered first

## Output Format

Respond with a JSON object in this exact format:
```json
{
  "reasoning_plan": "Brief description of the overall reasoning strategy",
  "estimated_hops": 3,
  "sub_questions": [
    {
      "id": "q1",
      "question": "First sub-question",
      "dependencies": []
    },
    {
      "id": "q2",
      "question": "Second sub-question",
      "dependencies": ["q1"]
    }
  ]
}
```

Ensure your response is valid JSON. The sub-questions should collectively enable answering the original question."""
    
    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for self-reflection."""
        return """You are performing Self-Reflection on an answer in a Multi-Agent RAG system. Evaluate the quality of the generated answer.

## Evaluation Criteria

1. **Completeness**: Does the answer address all parts of the question?
2. **Accuracy**: Is the answer factually correct based on the context?
3. **Coherence**: Is the answer well-structured and logical?
4. **Evidence Support**: Are claims supported by the provided context?

## Confidence Scoring

- **0.9-1.0**: Excellent answer, fully supported by evidence
- **0.7-0.9**: Good answer, minor issues or gaps
- **0.5-0.7**: Acceptable answer, significant gaps or uncertainties
- **0.0-0.5**: Poor answer, major issues or unsupported claims

## Output Format

Respond with a JSON object in this exact format:
```json
{
  "is_satisfactory": true,
  "confidence": 0.85,
  "issues": ["List any issues or gaps in the answer"],
  "suggestions": ["Suggestions for improvement if retry is needed"],
  "should_retry": false,
  "retry_strategy": "Description of how to retry if needed, or null"
}
```

Set `should_retry` to true if confidence is below 0.7 or significant issues exist. Provide specific suggestions for improvement."""
    
    def _parse_decomposition_response(self, response: str) -> DecompositionResult:
        """Parse the LLM decomposition response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            sub_questions = [
                SubQuestion(
                    id=sq.get("id", f"q{i+1}"),
                    question=sq["question"],
                    dependencies=sq.get("dependencies", []),
                )
                for i, sq in enumerate(data.get("sub_questions", []))
            ]
            
            return DecompositionResult(
                sub_questions=sub_questions,
                reasoning_plan=data.get("reasoning_plan", "No plan provided"),
                estimated_hops=data.get("estimated_hops", len(sub_questions)),
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: return original question as single sub-question
            return DecompositionResult(
                sub_questions=[SubQuestion(id="q1", question="Original question")],
                reasoning_plan=f"Parse error: {str(e)}. Using fallback.",
                estimated_hops=1,
            )
    
    def _parse_reflection_response(self, response: str) -> ReflectionResult:
        """Parse the LLM reflection response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            confidence = float(data.get("confidence", 0.5))
            is_satisfactory = data.get("is_satisfactory", confidence >= 0.7)
            should_retry = data.get("should_retry", confidence < 0.7)
            
            return ReflectionResult(
                is_satisfactory=is_satisfactory,
                confidence=confidence,
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                should_retry=should_retry,
                retry_strategy=data.get("retry_strategy"),
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: parse from keywords
            response_lower = response.lower()
            confidence = 0.5
            
            if "high" in response_lower or "good" in response_lower:
                confidence = 0.8
                is_satisfactory = True
                should_retry = False
            elif "low" in response_lower or "poor" in response_lower:
                confidence = 0.3
                is_satisfactory = False
                should_retry = True
            else:
                is_satisfactory = True
                should_retry = False
            
            return ReflectionResult(
                is_satisfactory=is_satisfactory,
                confidence=confidence,
                issues=[f"Parse error: {str(e)}"],
                suggestions=["Review manually"],
                should_retry=should_retry,
                retry_strategy=None,
            )


class ReasoningEvaluator:
    """
    Evaluator for Reasoning Agent performance.
    
    Provides metrics for:
    - Decomposition quality
    - Answer accuracy
    - Self-reflection calibration
    """
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        question: str,
        decomposition: DecompositionResult,
        output: AgentOutput,
        ground_truth: Optional[str] = None,
    ) -> None:
        """Add a result for evaluation."""
        self.results.append({
            "question": question,
            "decomposition": decomposition.to_dict(),
            "output": output.to_dict() if hasattr(output, "to_dict") else output,
            "ground_truth": ground_truth,
        })
    
    def evaluate_decomposition(self) -> Dict[str, Any]:
        """Evaluate decomposition quality."""
        if not self.results:
            return {"error": "No results to evaluate"}
        
        total_sub_qs = sum(
            len(r["decomposition"]["sub_questions"])
            for r in self.results
        )
        avg_sub_qs = total_sub_qs / len(self.results)
        
        return {
            "avg_sub_questions": avg_sub_qs,
            "total_decompositions": len(self.results),
        }
    
    def evaluate_reflection_calibration(self) -> Dict[str, Any]:
        """Evaluate how well self-reflection confidence matches actual accuracy."""
        if not self.results:
            return {"error": "No results to evaluate"}
        
        confidences = []
        for r in self.results:
            output = r.get("output", {})
            if isinstance(output, dict):
                confidences.append(output.get("confidence", 0.5))
        
        if not confidences:
            return {"error": "No confidence scores found"}
        
        import statistics
        return {
            "avg_confidence": statistics.mean(confidences),
            "std_confidence": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
        }
