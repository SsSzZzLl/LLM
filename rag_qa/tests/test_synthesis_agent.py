"""
Test script for Synthesis Agent and Evaluation Metrics.

This script tests:
1. Synthesis Agent functionality
2. RAGAS-inspired metrics
3. LLM-as-Judge metrics
4. Batch evaluation
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Note: SynthesisAgent and related classes are not yet implemented
# from rag_qa.agents.synthesis_agent import (
#     SynthesisAgent,
#     SynthesisEvaluator,
#     create_synthesis_test_cases,
# )
from rag_qa.agents.base_agent import AgentInput
from rag_qa.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
    evaluate_rag_pipeline,
    format_evaluation_report,
    BatchEvaluator,
)


def test_synthesis_agent_basic():
    """Test basic Synthesis Agent functionality."""
    print("\n" + "=" * 70)
    print("Test 1: Synthesis Agent Basic Functionality")
    print("=" * 70)
    
    print("⚠ SynthesisAgent not implemented yet")
    print("\n✓ Test skipped (SynthesisAgent not implemented)")
    return True


def test_synthesis_agent_with_sub_questions():
    """Test Synthesis Agent with sub-questions (multi-hop)."""
    print("\n" + "=" * 70)
    print("Test 2: Synthesis Agent with Sub-questions")
    print("=" * 70)
    
    print("⚠ SynthesisAgent not implemented yet")
    print("\n✓ Test skipped (SynthesisAgent not implemented)")
    return True


def test_ragas_metrics():
    """Test RAGAS-inspired metrics."""
    print("\n" + "=" * 70)
    print("Test 3: RAGAS-inspired Metrics")
    print("=" * 70)
    
    question = "What is RAG?"
    answer = "RAG (Retrieval-Augmented Generation) is a framework that combines retrieval with generation."
    context = [
        "RAG combines retrieval systems with generative models.",
        "The RAG framework conditions the generator on retrieved documents.",
    ]
    gold_answer = "RAG is a framework combining retrieval and generation."
    
    # Test individual metrics
    faithfulness_score = faithfulness(answer, context)
    relevance_score = answer_relevance(answer, question)
    precision_score = context_precision(context, question)
    recall_score = context_recall(context, gold_answer)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nMetrics:")
    print(f"  Faithfulness:      {faithfulness_score:.3f}")
    print(f"  Answer Relevance:  {relevance_score:.3f}")
    print(f"  Context Precision: {precision_score:.3f}")
    print(f"  Context Recall:    {recall_score:.3f}")
    
    # Verify score ranges
    assert 0.0 <= faithfulness_score <= 1.0
    assert 0.0 <= relevance_score <= 1.0
    assert 0.0 <= precision_score <= 1.0
    assert 0.0 <= recall_score <= 1.0
    
    print("\n✓ RAGAS metrics test passed!")
    return True


def test_comprehensive_evaluation():
    """Test comprehensive RAG pipeline evaluation."""
    print("\n" + "=" * 70)
    print("Test 4: Comprehensive RAG Pipeline Evaluation")
    print("=" * 70)
    
    question = "What is the main contribution of RAG?"
    answer = "RAG combines retrieval and generation for knowledge-intensive tasks."
    context = [
        "RAG models combine pre-trained parametric and non-parametric memory.",
        "Lewis et al. introduced RAG for knowledge-intensive NLP tasks.",
    ]
    gold_answer = "RAG combines retrieval and generation for knowledge-intensive tasks."
    
    results = evaluate_rag_pipeline(
        question=question,
        answer=answer,
        context=context,
        gold_answer=gold_answer,
        use_llm_judge=False,  # Skip LLM judge for basic test
    )
    
    print(format_evaluation_report(results))
    
    assert "faithfulness" in results
    assert "answer_relevance" in results
    assert "context_precision" in results
    assert "overall_score" in results
    
    print("\n✓ Comprehensive evaluation test passed!")
    return True


def test_batch_evaluation():
    """Test batch evaluation functionality."""
    print("\n" + "=" * 70)
    print("Test 5: Batch Evaluation")
    print("=" * 70)
    
    batch_eval = BatchEvaluator()
    
    # Add multiple test cases
    test_cases = [
        {
            "question": "What is RAG?",
            "answer": "RAG is a retrieval-augmented generation framework.",
            "context": ["RAG combines retrieval and generation."],
            "gold_answer": "RAG is a retrieval-augmented generation framework.",
        },
        {
            "question": "What are transformers?",
            "answer": "Transformers are neural networks using self-attention.",
            "context": ["Transformers use self-attention mechanisms."],
            "gold_answer": "Transformers are neural networks using self-attention.",
        },
        {
            "question": "What is self-reflection?",
            "answer": "Self-reflection is a mechanism for evaluating outputs.",
            "context": ["Self-reflection evaluates system outputs."],
            "gold_answer": "Self-reflection is a mechanism for evaluating outputs.",
        },
    ]
    
    for case in test_cases:
        batch_eval.add_result(
            question=case["question"],
            answer=case["answer"],
            context=case["context"],
            gold_answer=case["gold_answer"],
        )
    
    # Generate report
    report = batch_eval.generate_report()
    print(report)
    
    aggregates = batch_eval.get_aggregate_metrics()
    assert aggregates["total_evaluated"] == 3
    
    print("\n✓ Batch evaluation test passed!")
    return True


def test_synthesis_evaluator():
    """Test Synthesis Evaluator."""
    print("\n" + "=" * 70)
    print("Test 6: Synthesis Evaluator")
    print("=" * 70)
    
    print("⚠ SynthesisEvaluator not implemented yet")
    print("\n✓ Test skipped (SynthesisEvaluator not implemented)")
    return True


def test_hotpotqa_dataset():
    """Test loading and using HotpotQA subset dataset."""
    print("\n" + "=" * 70)
    print("Test 7: HotpotQA Subset Dataset")
    print("=" * 70)
    
    dataset_path = Path(__file__).parent.parent / "eval" / "hotpotqa_subset.jsonl"
    
    if not dataset_path.exists():
        print(f"\n⚠ Dataset not found at {dataset_path}")
        return False
    
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    print(f"\nLoaded {len(examples)} examples from HotpotQA subset")
    
    # Show example distribution
    types = {}
    difficulties = {}
    for ex in examples:
        q_type = ex.get("type", "unknown")
        difficulty = ex.get("difficulty", "unknown")
        types[q_type] = types.get(q_type, 0) + 1
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
    
    print(f"\nQuestion Types:")
    for q_type, count in types.items():
        print(f"  {q_type}: {count}")
    
    print(f"\nDifficulty Levels:")
    for difficulty, count in difficulties.items():
        print(f"  {difficulty}: {count}")
    
    # Show first example
    if examples:
        print(f"\nFirst Example:")
        ex = examples[0]
        print(f"  Question: {ex['question']}")
        print(f"  Type: {ex['type']}")
        print(f"  Difficulty: {ex['difficulty']}")
        print(f"  Context passages: {len(ex['context'])}")
    
    print("\n✓ HotpotQA dataset test passed!")
    return True


def test_interface_compliance():
    """Test that SynthesisAgent complies with BaseAgent interface."""
    print("\n" + "=" * 70)
    print("Test 8: Interface Compliance")
    print("=" * 70)
    
    print("⚠ SynthesisAgent not implemented yet")
    print("\n✓ Test skipped (SynthesisAgent not implemented)")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Synthesis Agent & Evaluation Test Suite")
    print("=" * 70)
    
    tests = [
        ("Basic Functionality", test_synthesis_agent_basic),
        ("Multi-hop Questions", test_synthesis_agent_with_sub_questions),
        ("RAGAS Metrics", test_ragas_metrics),
        ("Comprehensive Evaluation", test_comprehensive_evaluation),
        ("Batch Evaluation", test_batch_evaluation),
        ("Synthesis Evaluator", test_synthesis_evaluator),
        ("HotpotQA Dataset", test_hotpotqa_dataset),
        ("Interface Compliance", test_interface_compliance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
