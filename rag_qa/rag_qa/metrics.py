from __future__ import annotations

import re
import string
from typing import Optional, List, Dict, Any, Tuple


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\[[\d, ]+\]", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def token_f1(pred: str, gold: str) -> float:
    p = set(normalize_answer(pred).split())
    g = set(normalize_answer(gold).split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = p & g
    if not inter:
        return 0.0
    precision = len(inter) / len(p)
    recall = len(inter) / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_answer(pred: str, gold: str) -> bool:
    """Loose check: all gold tokens appear in pred (good for short spans)."""
    g = normalize_answer(gold).split()
    if not g:
        return True
    pn = normalize_answer(pred)
    return all(t in pn for t in g)


# ============================================================================
# RAGAS-inspired Metrics (Faithfulness, Relevance, Context Precision)
# ============================================================================

def faithfulness(answer: str, context: List[str]) -> float:
    """
    Measure how faithful the answer is to the retrieved context.
    
    This is a simplified implementation of RAGAS faithfulness metric.
    It checks if claims in the answer can be supported by the context.
    
    Args:
        answer: The generated answer
        context: List of retrieved passages
        
    Returns:
        Faithfulness score between 0.0 and 1.0
    """
    if not answer or not context:
        return 0.0
    
    # Normalize answer and context
    answer_normalized = normalize_answer(answer)
    context_text = " ".join([normalize_answer(c) for c in context])
    
    # Extract key phrases from answer (simple approach: noun phrases)
    answer_words = set(answer_normalized.split())
    context_words = set(context_text.split())
    
    # Calculate overlap (excluding common stop words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "and", "but", "or", "yet", "so", "if",
        "because", "although", "though", "while", "where", "when", "that",
        "which", "who", "whom", "whose", "what", "this", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their", "mine",
        "yours", "hers", "ours", "theirs", "myself", "yourself", "himself",
        "herself", "itself", "ourselves", "yourselves", "themselves"
    }
    
    # Filter out stop words
    answer_content = answer_words - stop_words
    context_content = context_words - stop_words
    
    if not answer_content:
        return 0.0  # Empty answer cannot be evaluated for faithfulness
    
    # Calculate proportion of answer content found in context
    supported = answer_content & context_content
    faithfulness_score = len(supported) / len(answer_content)
    
    return faithfulness_score


def answer_relevance(answer: str, question: str) -> float:
    """
    Measure how relevant the answer is to the question.
    
    This is a simplified implementation that checks semantic overlap
    between question and answer.
    
    Args:
        answer: The generated answer
        question: The original question
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    if not answer or not question:
        return 0.0
    
    # Normalize
    answer_normalized = normalize_answer(answer)
    question_normalized = normalize_answer(question)
    
    # Extract key terms (excluding stop words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "and", "but", "or", "yet", "so", "if",
        "because", "although", "though", "while", "where", "when", "that",
        "which", "who", "whom", "whose", "what", "this", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their", "mine",
        "yours", "hers", "ours", "theirs", "myself", "yourself", "himself",
        "herself", "itself", "ourselves", "yourselves", "themselves",
        "how", "why", "where", "when", "what", "who", "which", "whose", "whom"
    }
    
    answer_words = set(answer_normalized.split()) - stop_words
    question_words = set(question_normalized.split()) - stop_words
    
    if not question_words:
        return 1.0 if answer_words else 0.0
    
    # Calculate how many question terms appear in answer
    overlap = answer_words & question_words
    relevance_score = len(overlap) / len(question_words)
    
    # Boost score if answer has substantial content
    if len(answer_normalized.split()) < 5:
        relevance_score *= 0.5  # Penalize very short answers
    
    return min(relevance_score, 1.0)


def context_precision(context: List[str], question: str) -> float:
    """
    Measure the precision of retrieved context.
    
    Checks what proportion of retrieved passages are relevant to the question.
    
    Args:
        context: List of retrieved passages
        question: The original question
        
    Returns:
        Precision score between 0.0 and 1.0
    """
    if not context:
        return 0.0
    
    question_normalized = normalize_answer(question)
    question_words = set(question_normalized.split())
    
    # Common words to ignore
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "and", "but", "or", "yet", "so", "if",
        "because", "although", "though", "while", "where", "when", "that",
        "which", "who", "whom", "whose", "what", "this", "these", "those",
        "how", "why", "where", "when", "what", "who", "which", "whose", "whom"
    }
    
    question_content = question_words - stop_words
    
    if not question_content:
        return 0.0  # If question has no content words, cannot evaluate relevance
    
    relevant_count = 0
    for passage in context:
        passage_normalized = normalize_answer(passage)
        passage_words = set(passage_normalized.split())
        passage_content = passage_words - stop_words
        
        # Check if passage shares content with question
        overlap = passage_content & question_content
        if len(overlap) > 0:
            relevant_count += 1
    
    return relevant_count / len(context)


def context_recall(context: List[str], gold_answer: str) -> float:
    """
    Measure the recall of retrieved context with respect to gold answer.
    
    Checks what proportion of gold answer content is covered by the context.
    
    Args:
        context: List of retrieved passages
        gold_answer: Ground truth answer
        
    Returns:
        Recall score between 0.0 and 1.0
    """
    if not context or not gold_answer:
        return 0.0
    
    gold_normalized = normalize_answer(gold_answer)
    gold_words = set(gold_normalized.split())
    
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "and", "but", "or", "yet", "so", "if",
        "because", "although", "though", "while", "where", "when", "that",
        "which", "who", "whom", "whose", "what", "this", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their"
    }
    
    gold_content = gold_words - stop_words
    
    if not gold_content:
        return 0.0  # Empty gold answer cannot be evaluated for recall
    
    # Combine all context
    context_text = " ".join([normalize_answer(c) for c in context])
    context_words = set(context_text.split())
    context_content = context_words - stop_words
    
    # Calculate coverage
    covered = gold_content & context_content
    recall_score = len(covered) / len(gold_content)
    
    return recall_score


# ============================================================================
# LLM-as-Judge Metrics
# ============================================================================

class LLMJudgeEvaluator:
    """
    LLM-as-Judge evaluator for RAG quality assessment.
    
    Uses LLM to evaluate answer quality across multiple dimensions.
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM Judge.
        
        Args:
            model: LLM model name for evaluation
        """
        self.model = model
    
    def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        gold_answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate answer correctness using LLM.
        
        Args:
            question: Original question
            answer: Generated answer
            gold_answer: Ground truth answer
            
        Returns:
            Evaluation result with score and reasoning
        """
        from rag_qa.generate import generate_chat
        
        system_prompt = """You are an expert evaluator assessing the correctness of answers.

Evaluate the generated answer against the ground truth answer on a scale of 0-10, where:
- 10: Perfect match, all facts correct
- 7-9: Mostly correct with minor errors or omissions
- 4-6: Partially correct, significant errors or missing information
- 1-3: Mostly incorrect
- 0: Completely wrong or irrelevant

Provide your evaluation in JSON format:
{
  "score": 8,
  "reasoning": "Brief explanation of the score",
  "correct_facts": ["List correct facts"],
  "incorrect_facts": ["List incorrect or missing facts"]
}"""
        
        user_prompt = f"""Question: {question}

Generated Answer: {answer}

Ground Truth Answer: {gold_answer}

Please evaluate the correctness of the generated answer."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=512,
            )
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": result.get("score", 0) / 10.0,  # Normalize to 0-1
                    "reasoning": result.get("reasoning", ""),
                    "correct_facts": result.get("correct_facts", []),
                    "incorrect_facts": result.get("incorrect_facts", []),
                }
            else:
                return {
                    "score": 0.5,
                    "reasoning": "Failed to parse LLM response",
                    "correct_facts": [],
                    "incorrect_facts": [],
                }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
                "correct_facts": [],
                "incorrect_facts": [],
            }
    
    def evaluate_answer_helpfulness(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate how helpful the answer is.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Evaluation result with helpfulness score
        """
        from rag_qa.generate import generate_chat
        
        system_prompt = """You are an expert evaluator assessing the helpfulness of answers.

Rate the helpfulness of the answer on a scale of 0-10, considering:
- Does it directly address the question?
- Is it clear and easy to understand?
- Does it provide sufficient detail?
- Is it well-structured?

Provide your evaluation in JSON format:
{
  "score": 8,
  "reasoning": "Brief explanation",
  "strengths": ["What the answer does well"],
  "weaknesses": ["What could be improved"]
}"""
        
        user_prompt = f"""Question: {question}

Answer: {answer}

Please evaluate the helpfulness of this answer."""
        
        try:
            response = generate_chat(
                system=system_prompt,
                user=user_prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=512,
            )
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "score": result.get("score", 0) / 10.0,
                    "reasoning": result.get("reasoning", ""),
                    "strengths": result.get("strengths", []),
                    "weaknesses": result.get("weaknesses", []),
                }
            else:
                return {
                    "score": 0.5,
                    "reasoning": "Failed to parse response",
                    "strengths": [],
                    "weaknesses": [],
                }
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
                "strengths": [],
                "weaknesses": [],
            }


# ============================================================================
# Comprehensive RAG Evaluation Suite
# ============================================================================

def evaluate_rag_pipeline(
    question: str,
    answer: str,
    context: List[str],
    gold_answer: Optional[str] = None,
    use_llm_judge: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive RAG pipeline evaluation.
    
    Evaluates all aspects of the RAG pipeline output.
    
    Args:
        question: Original question
        answer: Generated answer
        context: Retrieved passages
        gold_answer: Ground truth answer (optional)
        use_llm_judge: Whether to use LLM-as-Judge
        model: Model for LLM judge
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {
        "question": question,
        "answer": answer,
        "context_count": len(context),
    }
    
    # RAGAS-inspired metrics
    results["faithfulness"] = faithfulness(answer, context)
    results["answer_relevance"] = answer_relevance(answer, question)
    results["context_precision"] = context_precision(context, question)
    
    if gold_answer:
        results["context_recall"] = context_recall(context, gold_answer)
        results["token_f1"] = token_f1(answer, gold_answer)
        results["exact_match"] = exact_match(answer, gold_answer)
        
        # LLM-as-Judge
        if use_llm_judge:
            judge = LLMJudgeEvaluator(model=model)
            correctness = judge.evaluate_answer_correctness(question, answer, gold_answer)
            results["llm_correctness"] = correctness["score"]
            results["llm_correctness_reasoning"] = correctness["reasoning"]
    
    # LLM-as-Judge for helpfulness (doesn't require gold answer)
    if use_llm_judge:
        judge = LLMJudgeEvaluator(model=model)
        helpfulness = judge.evaluate_answer_helpfulness(question, answer)
        results["llm_helpfulness"] = helpfulness["score"]
        results["llm_helpfulness_reasoning"] = helpfulness["reasoning"]
    
    # Calculate overall score
    base_metrics = [
        results.get("faithfulness", 0),
        results.get("answer_relevance", 0),
        results.get("context_precision", 0),
    ]
    
    if gold_answer:
        base_metrics.append(results.get("token_f1", 0))
    
    results["overall_score"] = sum(base_metrics) / len(base_metrics)
    
    return results


def format_evaluation_report(results: Dict[str, Any]) -> str:
    """
    Format evaluation results as a readable report.
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        "RAG Pipeline Evaluation Report",
        "=" * 70,
        "",
        f"Question: {results.get('question', 'N/A')}",
        f"Answer: {results.get('answer', 'N/A')[:100]}...",
        "",
        "Metrics:",
        "-" * 70,
    ]
    
    # Core metrics
    if "faithfulness" in results:
        lines.append(f"  Faithfulness:     {results['faithfulness']:.3f}")
    if "answer_relevance" in results:
        lines.append(f"  Answer Relevance: {results['answer_relevance']:.3f}")
    if "context_precision" in results:
        lines.append(f"  Context Precision:{results['context_precision']:.3f}")
    if "context_recall" in results:
        lines.append(f"  Context Recall:   {results['context_recall']:.3f}")
    
    # Traditional metrics
    if "token_f1" in results:
        lines.append(f"  Token F1:         {results['token_f1']:.3f}")
    if "exact_match" in results:
        lines.append(f"  Exact Match:      {results['exact_match']}")
    
    # LLM Judge metrics
    if "llm_correctness" in results:
        lines.append(f"  LLM Correctness:  {results['llm_correctness']:.3f}")
    if "llm_helpfulness" in results:
        lines.append(f"  LLM Helpfulness:  {results['llm_helpfulness']:.3f}")
    
    # Overall
    if "overall_score" in results:
        lines.append("-" * 70)
        lines.append(f"  Overall Score:    {results['overall_score']:.3f}")
    
    lines.extend([
        "=" * 70,
    ])
    
    return "\n".join(lines)


# ============================================================================
# Batch Evaluation Utilities
# ============================================================================

class BatchEvaluator:
    """
    Batch evaluator for running evaluations on multiple examples.
    """
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        question: str,
        answer: str,
        context: List[str],
        gold_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single example and add to batch.
        
        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved passages
            gold_answer: Ground truth answer
            
        Returns:
            Evaluation results
        """
        result = evaluate_rag_pipeline(
            question=question,
            answer=answer,
            context=context,
            gold_answer=gold_answer,
        )
        self.results.append(result)
        return result
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics across all evaluated examples.
        
        Returns:
            Dictionary with average metrics
        """
        if not self.results:
            return {"error": "No results to aggregate"}
        
        metrics_to_average = [
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
            "token_f1",
            "overall_score",
        ]
        
        aggregates = {}
        for metric in metrics_to_average:
            values = [r[metric] for r in self.results if metric in r]
            if values:
                aggregates[f"avg_{metric}"] = sum(values) / len(values)
        
        aggregates["total_evaluated"] = len(self.results)
        
        # Exact match accuracy
        em_values = [r["exact_match"] for r in self.results if "exact_match" in r]
        if em_values:
            aggregates["exact_match_accuracy"] = sum(em_values) / len(em_values)
        
        return aggregates
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Formatted report string
        """
        aggregates = self.get_aggregate_metrics()
        
        lines = [
            "=" * 70,
            "Batch Evaluation Report",
            "=" * 70,
            "",
            f"Total Examples Evaluated: {aggregates.get('total_evaluated', 0)}",
            "",
            "Aggregate Metrics:",
            "-" * 70,
        ]
        
        for key, value in aggregates.items():
            if key != "total_evaluated":
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset the evaluator."""
        self.results = []


if __name__ == "__main__":
    # Demo: Test the new metrics
    print("=" * 70)
    print("RAG Metrics Demo")
    print("=" * 70)
    
    # Test data
    question = "What is RAG?"
    answer = "RAG (Retrieval-Augmented Generation) is a framework that combines retrieval with generation."
    context = [
        "RAG combines retrieval systems with generative models.",
        "The RAG framework conditions the generator on retrieved documents.",
    ]
    gold_answer = "RAG is a framework combining retrieval and generation."
    
    # Test individual metrics
    print("\nIndividual Metrics:")
    print(f"  Faithfulness: {faithfulness(answer, context):.3f}")
    print(f"  Answer Relevance: {answer_relevance(answer, question):.3f}")
    print(f"  Context Precision: {context_precision(context, question):.3f}")
    print(f"  Context Recall: {context_recall(context, gold_answer):.3f}")
    print(f"  Token F1: {token_f1(answer, gold_answer):.3f}")
    
    # Test comprehensive evaluation
    print("\nComprehensive Evaluation:")
    results = evaluate_rag_pipeline(
        question=question,
        answer=answer,
        context=context,
        gold_answer=gold_answer,
    )
    print(format_evaluation_report(results))
    
    # Test batch evaluation
    print("\nBatch Evaluation:")
    batch_eval = BatchEvaluator()
    batch_eval.add_result(question, answer, context, gold_answer)
    batch_eval.add_result(
        "What is a transformer?",
        "Transformers are neural network architectures using self-attention.",
        ["Transformers use self-attention mechanisms.", "They were introduced in 2017."],
        "Transformers are neural networks using self-attention.",
    )
    print(batch_eval.generate_report())
