"""
Route Agent - Dynamic Routing for Multi-Agent RAG System

This module implements the RouteAgent which uses LLM to classify question complexity
and route questions to appropriate processing strategies.

Author: 耿源 (Route Agent Developer)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from rag_qa.generate import generate_chat


class QuestionComplexity(Enum):
    """Question complexity levels for routing decisions."""
    SIMPLE = "simple"           # Can be answered directly without retrieval
    MODERATE = "moderate"       # Requires single-hop retrieval
    COMPLEX = "complex"         # Requires multi-hop reasoning


@dataclass
class RouteDecision:
    """Routing decision containing complexity classification and strategy."""
    complexity: QuestionComplexity
    confidence: float           # Confidence score (0.0 - 1.0)
    reasoning: str              # Explanation for the routing decision
    recommended_strategy: str   # Specific strategy recommendation
    telemetry: Dict[str, Any] = None # Tracks agent latency and tokens
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommended_strategy": self.recommended_strategy,
            "telemetry": self.telemetry or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteDecision":
        return cls(
            complexity=QuestionComplexity(data.get("complexity", "moderate")),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "")),
            recommended_strategy=str(data.get("recommended_strategy", "")),
            telemetry=data.get("telemetry", {}),
        )


class RouteAgent:
    """
    Route Agent for dynamic question routing in Multi-Agent RAG system.
    
    Uses LLM to classify question complexity and determine the optimal
    processing strategy:
    - SIMPLE: Direct generation without retrieval
    - MODERATE: Single-hop retrieval + generation
    - COMPLEX: Multi-hop reasoning with iterative retrieval
    """
    
    # Routing strategies mapped to complexity levels
    ROUTING_STRATEGIES = {
        QuestionComplexity.SIMPLE: {
            "name": "direct_generation",
            "description": "Answer directly using LLM knowledge without retrieval",
            "use_retrieval": False,
            "use_multi_hop": False,
        },
        QuestionComplexity.MODERATE: {
            "name": "single_hop_rag",
            "description": "Single-hop retrieval followed by answer generation",
            "use_retrieval": True,
            "use_multi_hop": False,
        },
        QuestionComplexity.COMPLEX: {
            "name": "multi_hop_reasoning",
            "description": "Multi-hop reasoning with iterative retrieval and decomposition",
            "use_retrieval": True,
            "use_multi_hop": True,
        },
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Initialize RouteAgent.
        
        Args:
            model: LLM model name for classification
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens for classification response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def classify(self, question: str) -> RouteDecision:
        """
        Classify question complexity using LLM.
        
        Args:
            question: The user question to classify
            
        Returns:
            RouteDecision containing complexity and routing strategy
        """
        system_prompt = self._get_classification_prompt()
        user_prompt = f"Question: {question}\n\nProvide your classification in the requested JSON format."
        tracker = {"agent_name": "RouteAgent"}
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tracker=tracker,
            )
            
            # Parse the JSON response
            decision = self._parse_classification_response(response)
            decision.telemetry = tracker
            return decision
            
        except Exception as e:
            # Fallback to moderate complexity on error
            return RouteDecision(
                complexity=QuestionComplexity.MODERATE,
                confidence=0.5,
                reasoning=f"Classification failed: {str(e)}. Defaulting to moderate complexity.",
                recommended_strategy=self.ROUTING_STRATEGIES[QuestionComplexity.MODERATE]["name"],
                telemetry=tracker,
            )
    
    def route(self, question: str) -> Tuple[RouteDecision, Dict[str, Any]]:
        """
        Route a question and return the complete routing configuration.
        
        Args:
            question: The user question to route
            
        Returns:
            Tuple of (RouteDecision, routing_configuration)
        """
        decision = self.classify(question)
        strategy = self.ROUTING_STRATEGIES[decision.complexity]
        
        routing_config = {
            "strategy_name": strategy["name"],
            "strategy_description": strategy["description"],
            "use_retrieval": strategy["use_retrieval"],
            "use_multi_hop": strategy["use_multi_hop"],
            "complexity_score": decision.confidence,
        }
        
        return decision, routing_config
    
    def batch_classify(self, questions: List[str]) -> List[RouteDecision]:
        """
        Classify multiple questions in batch.
        
        Args:
            questions: List of questions to classify
            
        Returns:
            List of RouteDecision objects
        """
        return [self.classify(q) for q in questions]
    
    def _get_classification_prompt(self) -> str:
        """Get the system prompt for question complexity classification."""
        return """You are a Route Agent in a Multi-Agent RAG system. Your task is to classify questions by complexity and determine the optimal processing strategy.

## Complexity Levels

1. **SIMPLE**: Questions that can be answered directly using general knowledge without retrieval.
   - Examples: "What is 2+2?", "Define machine learning", "Who wrote Romeo and Juliet?"
   - Characteristics: Factual, well-known, single concept, no domain-specific context needed

2. **MODERATE**: Questions requiring single-hop retrieval from a document collection.
   - Examples: "What does RAG condition the generator on?", "What architecture did Vaswani et al. propose?"
   - Characteristics: Need specific information from documents, single information source, direct answer exists

3. **COMPLEX**: Questions requiring multi-hop reasoning or connecting multiple pieces of information. This includes "Bridging" or "Intersection" questions where uncovering one entity is required to search for another.
   - Examples: "Were Scott Derrickson and Ed Wood of the same nationality?", "What government position was held by the woman who portrayed X in the film Y?", "Compare the approaches of Lewis and Vaswani."
   - Characteristics: Multiple reasoning steps, comparison/synthesis required, implicit connections, bridging via an intermediate entity.

## Classification Guidelines

- Consider if the question requires information from external documents
- Check if multiple pieces of information need to be connected
- Evaluate if the answer requires synthesis or comparison
- Assess confidence based on clarity of the question

## Output Format

Respond with a JSON object in this exact format:
```json
{
  "complexity": "simple|moderate|complex",
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this complexity level was chosen",
  "recommended_strategy": "Name of the recommended strategy"
}
```

Ensure your response is valid JSON. Be decisive in your classification."""
    
    def _parse_classification_response(self, response: str) -> RouteDecision:
        """Parse the LLM classification response into a RouteDecision."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
            else:
                # Try parsing the entire response as JSON
                data = json.loads(response)
            
            complexity_str = data.get("complexity", "moderate").lower()
            complexity = QuestionComplexity(complexity_str)
            
            return RouteDecision(
                complexity=complexity,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "No reasoning provided")),
                recommended_strategy=str(data.get("recommended_strategy", self.ROUTING_STRATEGIES[complexity]["name"])),
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to infer from keywords
            response_lower = response.lower()
            if "simple" in response_lower:
                complexity = QuestionComplexity.SIMPLE
            elif "complex" in response_lower:
                complexity = QuestionComplexity.COMPLEX
            else:
                complexity = QuestionComplexity.MODERATE
            
            return RouteDecision(
                complexity=complexity,
                confidence=0.6,
                reasoning=f"Parsed from keyword matching. Original parse error: {str(e)}",
                recommended_strategy=self.ROUTING_STRATEGIES[complexity]["name"],
            )


class RouteEvaluator:
    """
    Evaluator for Route Agent classification accuracy.
    
    Provides metrics for assessing routing quality:
    - Classification accuracy
    - Confusion matrix
    - Confidence calibration
    """
    
    def __init__(self):
        self.predictions: List[RouteDecision] = []
        self.ground_truth: List[QuestionComplexity] = []
        self.questions: List[str] = []
    
    def add_prediction(
        self,
        question: str,
        prediction: RouteDecision,
        ground_truth: QuestionComplexity,
    ) -> None:
        """Add a prediction for evaluation."""
        self.questions.append(question)
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate routing accuracy and return metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.predictions:
            return {"error": "No predictions to evaluate"}
        
        # Calculate accuracy
        correct = sum(
            1 for pred, gt in zip(self.predictions, self.ground_truth)
            if pred.complexity == gt
        )
        accuracy = correct / len(self.predictions)
        
        # Calculate per-class metrics
        classes = list(QuestionComplexity)
        per_class_metrics = {}
        
        for cls in classes:
            cls_preds = [p for p in self.predictions if p.complexity == cls]
            cls_gt = [g for g in self.ground_truth if g == cls]
            
            true_positives = sum(
                1 for p, g in zip(self.predictions, self.ground_truth)
                if p.complexity == cls and g == cls
            )
            false_positives = sum(
                1 for p, g in zip(self.predictions, self.ground_truth)
                if p.complexity == cls and g != cls
            )
            false_negatives = sum(
                1 for p, g in zip(self.predictions, self.ground_truth)
                if p.complexity != cls and g == cls
            )
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[cls.value] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": len(cls_gt),
            }
        
        # Confusion matrix
        confusion_matrix = {}
        for pred_cls in classes:
            confusion_matrix[pred_cls.value] = {}
            for gt_cls in classes:
                count = sum(
                    1 for p, g in zip(self.predictions, self.ground_truth)
                    if p.complexity == pred_cls and g == gt_cls
                )
                confusion_matrix[pred_cls.value][gt_cls.value] = count
        
        # Average confidence
        avg_confidence = sum(p.confidence for p in self.predictions) / len(self.predictions)
        
        return {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "total_samples": len(self.predictions),
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix,
        }
    
    def get_misclassified(self) -> List[Dict[str, Any]]:
        """Get list of misclassified examples for analysis."""
        misclassified = []
        for question, pred, gt in zip(self.questions, self.predictions, self.ground_truth):
            if pred.complexity != gt:
                misclassified.append({
                    "question": question,
                    "predicted": pred.complexity.value,
                    "ground_truth": gt.value,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning,
                })
        return misclassified
    
    def reset(self) -> None:
        """Reset the evaluator state."""
        self.predictions = []
        self.ground_truth = []
        self.questions = []


def create_test_dataset() -> List[Tuple[str, QuestionComplexity]]:
    """
    Create a test dataset for evaluating Route Agent.
    
    Returns:
        List of (question, complexity) tuples
    """
    return [
        # SIMPLE questions
        ("What is 2+2?", QuestionComplexity.SIMPLE),
        ("Define machine learning.", QuestionComplexity.SIMPLE),
        ("Who wrote Romeo and Juliet?", QuestionComplexity.SIMPLE),
        ("What is the capital of France?", QuestionComplexity.SIMPLE),
        ("Explain what is a neural network.", QuestionComplexity.SIMPLE),
        
        # MODERATE questions
        ("What does RAG condition the generator on?", QuestionComplexity.MODERATE),
        ("Who introduced RAG for knowledge-intensive tasks?", QuestionComplexity.MODERATE),
        ("What architecture did Vaswani et al. propose in 2017?", QuestionComplexity.MODERATE),
        ("What is hallucination in the context of LLM evaluation?", QuestionComplexity.MODERATE),
        ("What should teams document for reproducibility?", QuestionComplexity.MODERATE),
        
        # COMPLEX questions
        ("Compare the RAG approach by Lewis et al. with the transformer architecture by Vaswani et al. for knowledge-intensive tasks.", QuestionComplexity.COMPLEX),
        ("How does the hallucination problem in LLMs relate to the retrieval mechanism in RAG systems, and what evaluation metrics can address both?", QuestionComplexity.COMPLEX),
        ("What are the trade-offs between using dense retrieval versus BM25 in multi-hop question answering scenarios?", QuestionComplexity.COMPLEX),
        ("Explain how self-reflection mechanisms in agent systems could improve the retrieval quality in RAG pipelines.", QuestionComplexity.COMPLEX),
        ("What would be the implications of combining the chunking strategies from the course notes with the evaluation framework for reproducibility?", QuestionComplexity.COMPLEX),
    ]


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Route Agent Demo")
    print("=" * 60)
    
    agent = RouteAgent()
    
    test_questions = [
        "What is the meaning of life?",
        "What does RAG condition the generator on?",
        "Compare and contrast the transformer architecture with RAG for knowledge-intensive NLP tasks.",
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        decision, config = agent.route(question)
        print(f"  Complexity: {decision.complexity.value}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Strategy: {config['strategy_name']}")
        print(f"  Reasoning: {decision.reasoning}")
