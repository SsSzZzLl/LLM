"""
Synthesis Agent - Multi-Agent RAG System

This module implements the SynthesisAgent which integrates evidence from multiple
sources and generates the final answer with self-reflection capabilities.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag_qa.agents.base_agent import AgentInput, AgentOutput, BaseAgent
from rag_qa.generate import generate_chat


@dataclass
class SynthesisContext:
    """Context information for synthesis process."""
    retrieved_passages: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    intermediate_answers: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Result of self-reflection process."""
    is_satisfactory: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    should_retry: bool


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent for integrating evidence and generating final answers.
    
    Key responsibilities:
    1. Integrate evidence from multiple retrieved passages
    2. Resolve contradictions between sources
    3. Generate coherent final answer with citations
    4. Self-reflection: evaluate answer quality and decide whether to retry
    
    Implements the BaseAgent interface for compatibility with the multi-agent system.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_reflection_iterations: int = 2,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize SynthesisAgent.
        
        Args:
            model: LLM model name for synthesis
            temperature: Sampling temperature
            max_tokens: Maximum tokens for answer generation
            max_reflection_iterations: Maximum number of self-reflection iterations
            confidence_threshold: Minimum confidence score to accept answer
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_reflection_iterations = max_reflection_iterations
        self.confidence_threshold = confidence_threshold
    
    def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the synthesis process.
        
        This is the main entry point required by BaseAgent interface.
        
        Args:
            input_data: AgentInput containing question, context, sub_questions, metadata
            
        Returns:
            AgentOutput with answer, evidence, confidence, and should_retry flag
        """
        # Extract information from input
        question = input_data.question
        context = input_data.context
        sub_questions = input_data.sub_questions
        metadata = input_data.metadata
        
        # Build synthesis context
        synthesis_context = SynthesisContext(
            retrieved_passages=context,
            reasoning_steps=[],
            intermediate_answers=[],
            contradictions=[],
            confidence_scores=[],
        )
        
        # Perform synthesis with self-reflection
        final_answer, evidence, confidence, should_retry = self._synthesize_with_reflection(
            question=question,
            context=synthesis_context,
            sub_questions=sub_questions,
            metadata=metadata,
        )
        
        return AgentOutput(
            answer=final_answer,
            evidence=evidence,
            confidence=confidence,
            should_retry=should_retry,
            metadata={
                "synthesis_iterations": len(synthesis_context.reasoning_steps),
                "contradictions_found": len(synthesis_context.contradictions),
                "confidence_history": synthesis_context.confidence_scores,
            },
        )
    
    def _synthesize_with_reflection(
        self,
        question: str,
        context: SynthesisContext,
        sub_questions: List[str],
        metadata: Dict[str, Any],
    ) -> Tuple[str, List[str], float, bool]:
        """
        Perform synthesis with iterative self-reflection.
        
        Args:
            question: The original question
            context: Synthesis context with passages and reasoning
            sub_questions: Decomposed sub-questions (if any)
            metadata: Additional metadata
            
        Returns:
            Tuple of (final_answer, evidence, confidence, should_retry)
        """
        current_iteration = 0
        best_answer = ""
        best_evidence = []
        best_confidence = 0.0
        
        while current_iteration < self.max_reflection_iterations:
            # Generate answer
            answer, evidence = self._generate_answer(
                question=question,
                passages=context.retrieved_passages,
                sub_questions=sub_questions,
                previous_attempts=context.intermediate_answers,
            )
            
            # Perform self-reflection
            reflection = self._reflect(
                question=question,
                answer=answer,
                evidence=evidence,
                passages=context.retrieved_passages,
            )
            
            # Update context
            context.intermediate_answers.append(answer)
            context.confidence_scores.append(reflection.confidence)
            
            # Track best answer
            if reflection.confidence > best_confidence:
                best_answer = answer
                best_evidence = evidence
                best_confidence = reflection.confidence
            
            # Check if satisfactory
            if reflection.is_satisfactory and reflection.confidence >= self.confidence_threshold:
                return answer, evidence, reflection.confidence, False
            
            # If not satisfactory and we can retry, update context for next iteration
            if reflection.should_retry and current_iteration < self.max_reflection_iterations - 1:
                context.contradictions.extend(reflection.issues)
                # Update metadata with reflection feedback
                metadata["reflection_feedback"] = reflection.suggestions
            else:
                # Cannot or should not retry further
                break
            
            current_iteration += 1
        
        # Return best answer found
        should_retry = best_confidence < self.confidence_threshold
        return best_answer, best_evidence, best_confidence, should_retry
    
    def _generate_answer(
        self,
        question: str,
        passages: List[str],
        sub_questions: List[str],
        previous_attempts: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Generate answer by synthesizing evidence from passages.
        
        Args:
            question: The question to answer
            passages: Retrieved passages
            sub_questions: Sub-questions (if multi-hop)
            previous_attempts: Previous answer attempts (for retry)
            
        Returns:
            Tuple of (answer, evidence_list)
        """
        system_prompt = self._get_synthesis_prompt()
        
        # Build user prompt
        user_prompt = self._build_synthesis_user_prompt(
            question=question,
            passages=passages,
            sub_questions=sub_questions,
            previous_attempts=previous_attempts,
        )
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Parse response to extract answer and evidence
            answer, evidence = self._parse_synthesis_response(response, passages)
            return answer, evidence
            
        except Exception as e:
            # Fallback: return error message with empty evidence
            return f"[Synthesis Error: {str(e)}]", []
    
    def _reflect(
        self,
        question: str,
        answer: str,
        evidence: List[str],
        passages: List[str],
    ) -> ReflectionResult:
        """
        Perform self-reflection on the generated answer.
        
        Evaluates:
        - Answer completeness
        - Evidence support
        - Potential contradictions
        - Confidence in correctness
        
        Args:
            question: Original question
            answer: Generated answer
            evidence: Evidence used
            passages: All retrieved passages
            
        Returns:
            ReflectionResult with evaluation
        """
        system_prompt = self._get_reflection_prompt()
        
        user_prompt = f"""Question: {question}

Generated Answer: {answer}

Evidence Used:
{self._format_evidence(evidence)}

All Available Passages:
{self._format_passages(passages)}

Evaluate the answer quality and provide your assessment in JSON format."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=512,
            )
            
            return self._parse_reflection_response(response)
            
        except Exception as e:
            # Fallback: assume not satisfactory
            return ReflectionResult(
                is_satisfactory=False,
                confidence=0.5,
                issues=[f"Reflection error: {str(e)}"],
                suggestions=["Retry with different approach"],
                should_retry=True,
            )
    
    def _get_synthesis_prompt(self) -> str:
        """Get the system prompt for answer synthesis."""
        return """You are a Synthesis Agent in a Multi-Agent RAG system. Your task is to integrate evidence from multiple sources and generate a comprehensive, accurate answer.

## Your Responsibilities

1. **Evidence Integration**: Synthesize information from all provided passages
2. **Contradiction Resolution**: Identify and resolve contradictions between sources
3. **Citation**: Cite the passages you use (by their index numbers)
4. **Completeness**: Ensure all aspects of the question are addressed
5. **Clarity**: Present the answer in a clear, well-structured format

## Guidelines

- Use only the information from the provided passages
- If passages contain conflicting information, acknowledge it and explain your resolution
- Cite passages using [1], [2], etc. format
- If the information is insufficient, state this clearly
- For multi-hop questions, show the reasoning chain

## Output Format

Provide your answer in this format:

**Answer:**
[Your synthesized answer with citations]

**Evidence Used:**
[List the indices of passages you cited, e.g., 1, 3, 5]

**Reasoning:**
[Brief explanation of how you synthesized the information]"""
    
    def _get_reflection_prompt(self) -> str:
        """Get the system prompt for self-reflection."""
        return """You are a Self-Reflection module for a Synthesis Agent. Your task is to critically evaluate the quality of a generated answer.

## Evaluation Criteria

1. **Completeness**: Does the answer fully address the question?
2. **Accuracy**: Is the answer supported by the evidence?
3. **Consistency**: Are there internal contradictions?
4. **Citation Quality**: Are sources properly cited?
5. **Confidence**: How confident are you in this answer? (0.0-1.0)

## Output Format

Respond with a JSON object in this exact format:

```json
{
  "is_satisfactory": true/false,
  "confidence": 0.85,
  "issues": ["List any problems or concerns"],
  "suggestions": ["Suggestions for improvement if retry is needed"],
  "should_retry": true/false
}
```

- `is_satisfactory`: Whether the answer is good enough to return
- `confidence`: Your confidence score (0.0 to 1.0)
- `issues`: List of specific problems found
- `suggestions`: How to improve if retrying
- `should_retry`: Whether the system should try to generate a better answer"""
    
    def _build_synthesis_user_prompt(
        self,
        question: str,
        passages: List[str],
        sub_questions: List[str],
        previous_attempts: List[str],
    ) -> str:
        """Build the user prompt for synthesis."""
        prompt_parts = [f"Question: {question}\n"]
        
        # Add sub-questions if present
        if sub_questions:
            prompt_parts.append("Sub-questions to address:")
            for i, sq in enumerate(sub_questions, 1):
                prompt_parts.append(f"  {i}. {sq}")
            prompt_parts.append("")
        
        # Add passages
        prompt_parts.append("Retrieved Passages:")
        prompt_parts.append(self._format_passages(passages))
        prompt_parts.append("")
        
        # Add previous attempts if retrying
        if previous_attempts:
            prompt_parts.append("Previous Answer Attempts (avoid same mistakes):")
            for i, attempt in enumerate(previous_attempts, 1):
                prompt_parts.append(f"Attempt {i}: {attempt[:200]}...")
            prompt_parts.append("")
        
        prompt_parts.append("Please provide your synthesized answer following the format in the system instructions.")
        
        return "\n".join(prompt_parts)
    
    def _format_passages(self, passages: List[str]) -> str:
        """Format passages for prompt."""
        formatted = []
        for i, passage in enumerate(passages, 1):
            # Truncate long passages
            truncated = passage[:500] + "..." if len(passage) > 500 else passage
            formatted.append(f"[{i}] {truncated}")
        return "\n\n".join(formatted)
    
    def _format_evidence(self, evidence: List[str]) -> str:
        """Format evidence for prompt."""
        if not evidence:
            return "No specific evidence cited."
        return "\n".join(f"- {e[:200]}..." for e in evidence)
    
    def _parse_synthesis_response(self, response: str, passages: List[str]) -> Tuple[str, List[str]]:
        """Parse the synthesis response to extract answer and evidence."""
        answer = response
        evidence = []
        
        # Try to extract evidence indices
        evidence_pattern = r"\*\*Evidence Used:\*\*\s*([\d,\s]+)"
        evidence_match = re.search(evidence_pattern, response)
        
        if evidence_match:
            # Extract indices
            indices_str = evidence_match.group(1)
            indices = [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]
            
            # Map indices to passages
            for idx in indices:
                if 1 <= idx <= len(passages):
                    evidence.append(passages[idx - 1])
        else:
            # Fallback: extract cited passages from answer text
            citation_pattern = r"\[(\d+)\]"
            citations = re.findall(citation_pattern, response)
            indices = [int(c) for c in set(citations)]
            for idx in indices:
                if 1 <= idx <= len(passages):
                    evidence.append(passages[idx - 1])
        
        # Clean up the answer (remove the structured sections if present)
        answer = re.sub(r"\*\*Evidence Used:\*\*.*?(?=\n\n|\Z)", "", answer, flags=re.DOTALL)
        answer = re.sub(r"\*\*Reasoning:\*\*.*?(?=\n\n|\Z)", "", answer, flags=re.DOTALL)
        answer = re.sub(r"\*\*Answer:\*\*", "", answer)
        answer = answer.strip()
        
        return answer, evidence
    
    def _parse_reflection_response(self, response: str) -> ReflectionResult:
        """Parse the reflection response."""
        try:
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            return ReflectionResult(
                is_satisfactory=data.get("is_satisfactory", False),
                confidence=float(data.get("confidence", 0.5)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                should_retry=data.get("should_retry", False),
            )
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: parse from keywords
            response_lower = response.lower()
            is_satisfactory = "satisfactory" in response_lower and "not" not in response_lower.split("satisfactory")[0][-20:]
            confidence = 0.7 if is_satisfactory else 0.5
            
            return ReflectionResult(
                is_satisfactory=is_satisfactory,
                confidence=confidence,
                issues=["Failed to parse reflection JSON"],
                suggestions=["Manual review needed"],
                should_retry=not is_satisfactory,
            )


class SynthesisEvaluator:
    """
    Evaluator for Synthesis Agent output quality.
    
    Provides metrics for:
    - Answer completeness
    - Citation accuracy
    - Faithfulness to sources
    """
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        evidence: List[str],
        gold_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a synthesized answer.
        
        Args:
            question: Original question
            answer: Generated answer
            evidence: Evidence used
            gold_answer: Ground truth answer (if available)
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            "question": question,
            "answer_length": len(answer.split()),
            "evidence_count": len(evidence),
            "has_citations": "[" in answer and "]" in answer,
        }
        
        # Calculate coverage (how many evidence passages were used)
        if evidence:
            metrics["evidence_coverage"] = len(set(evidence)) / len(evidence)
        
        # Compare with gold answer if available
        if gold_answer:
            from rag_qa.metrics import token_f1, exact_match
            metrics["token_f1"] = token_f1(answer, gold_answer)
            metrics["exact_match"] = exact_match(answer, gold_answer)
        
        self.results.append(metrics)
        return metrics
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all evaluations."""
        if not self.results:
            return {"error": "No evaluations performed"}
        
        total = len(self.results)
        avg_answer_length = sum(r["answer_length"] for r in self.results) / total
        avg_evidence_count = sum(r["evidence_count"] for r in self.results) / total
        citation_rate = sum(1 for r in self.results if r["has_citations"]) / total
        
        result = {
            "total_evaluations": total,
            "avg_answer_length": avg_answer_length,
            "avg_evidence_count": avg_evidence_count,
            "citation_rate": citation_rate,
        }
        
        # Include F1 scores if available
        f1_scores = [r["token_f1"] for r in self.results if "token_f1" in r]
        if f1_scores:
            result["avg_token_f1"] = sum(f1_scores) / len(f1_scores)
        
        return result
    
    def reset(self) -> None:
        """Reset evaluator state."""
        self.results = []


def create_synthesis_test_cases() -> List[Dict[str, Any]]:
    """
    Create test cases for Synthesis Agent evaluation.
    
    Returns:
        List of test case dictionaries
    """
    return [
        {
            "question": "What is RAG and how does it work?",
            "passages": [
                "RAG (Retrieval-Augmented Generation) is a framework that combines retrieval systems with generative models.",
                "The RAG framework conditions the generator on retrieved documents to improve factual accuracy.",
                "RAG was introduced by Lewis et al. for knowledge-intensive NLP tasks.",
            ],
            "sub_questions": [],
            "gold_answer": "RAG (Retrieval-Augmented Generation) is a framework that combines retrieval systems with generative models, conditioning the generator on retrieved documents to improve factual accuracy.",
        },
        {
            "question": "Compare transformer and RAG approaches.",
            "passages": [
                "The transformer architecture was introduced by Vaswani et al. in 2017 for sequence transduction.",
                "Transformers use self-attention mechanisms to process sequences in parallel.",
                "RAG combines retrieval with generation, while transformers are pure generative models.",
                "Both approaches can be used for knowledge-intensive tasks but with different strengths.",
            ],
            "sub_questions": ["What is the transformer architecture?", "What is RAG?", "How do they differ?"],
            "gold_answer": None,
        },
        {
            "question": "What are the key components of a multi-agent RAG system?",
            "passages": [
                "Multi-agent RAG systems use specialized agents for different tasks.",
                "The Route Agent classifies question complexity.",
                "The Retrieval Agent handles document retrieval.",
                "The Reasoning Agent performs multi-hop reasoning.",
                "The Synthesis Agent integrates evidence and generates answers.",
            ],
            "sub_questions": [],
            "gold_answer": "A multi-agent RAG system consists of specialized agents including: Route Agent for question classification, Retrieval Agent for document retrieval, Reasoning Agent for multi-hop reasoning, and Synthesis Agent for evidence integration and answer generation.",
        },
    ]


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("Synthesis Agent Demo")
    print("=" * 70)
    
    agent = SynthesisAgent()
    
    # Test case 1: Simple synthesis
    print("\n[Test 1] Simple Question")
    print("-" * 70)
    
    input1 = AgentInput(
        question="What is RAG?",
        context=[
            "RAG (Retrieval-Augmented Generation) is a framework for knowledge-intensive tasks.",
            "RAG combines retrieval systems with generative models.",
        ],
        sub_questions=[],
        metadata={},
    )
    
    output1 = agent.run(input1)
    print(f"Question: {input1.question}")
    print(f"Answer: {output1.answer[:200]}...")
    print(f"Confidence: {output1.confidence:.2f}")
    print(f"Should Retry: {output1.should_retry}")
    print(f"Evidence Count: {len(output1.evidence)}")
    
    # Test case 2: Multi-hop question
    print("\n[Test 2] Multi-hop Question")
    print("-" * 70)
    
    input2 = AgentInput(
        question="How does RAG improve upon standard transformers?",
        context=[
            "Transformers are generative models that may hallucinate facts.",
            "RAG retrieves relevant documents before generation.",
            "RAG reduces hallucination by grounding generation in retrieved facts.",
        ],
        sub_questions=["What are transformers?", "What is RAG?", "How does retrieval help?"],
        metadata={},
    )
    
    output2 = agent.run(input2)
    print(f"Question: {input2.question}")
    print(f"Answer: {output2.answer[:200]}...")
    print(f"Confidence: {output2.confidence:.2f}")
    print(f"Should Retry: {output2.should_retry}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)
