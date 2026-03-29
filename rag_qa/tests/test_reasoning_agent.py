"""
Test script for Reasoning Agent.

This script tests the ReasoningAgent's ability to:
1. Decompose multi-hop questions
2. Perform self-reflection
3. Generate answers with confidence scores

Author: 叶子冉
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_qa.agents import ReasoningAgent, AgentInput


def test_basic_decomposition():
    """Test basic question decomposition."""
    print("=" * 60)
    print("Test 1: Basic Question Decomposition")
    print("=" * 60)
    
    agent = ReasoningAgent()
    
    question = "What is the relationship between RAG and transformer architecture?"
    context = [
        "RAG (Retrieval-Augmented Generation) combines retrieval systems with generation models.",
        "Transformer architecture was introduced by Vaswani et al. in 2017.",
        "RAG uses transformer-based models as its generator component.",
    ]
    
    input_data = AgentInput(
        question=question,
        context=context,
    )
    
    # Test decomposition
    decomposition = agent.decompose_question(question, context)
    
    print(f"Original Question: {question}")
    print(f"\nReasoning Plan: {decomposition.reasoning_plan}")
    print(f"Estimated Hops: {decomposition.estimated_hops}")
    print(f"\nSub-questions:")
    for sq in decomposition.sub_questions:
        print(f"  [{sq.id}] {sq.question}")
        if sq.dependencies:
            print(f"       Dependencies: {sq.dependencies}")
    
    print("\n✓ Decomposition test completed\n")
    return decomposition


def test_self_reflection():
    """Test self-reflection mechanism."""
    print("=" * 60)
    print("Test 2: Self-Reflection")
    print("=" * 60)
    
    agent = ReasoningAgent()
    
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."
    context = ["Paris is the capital and most populous city of France."]
    
    reflection = agent.reflect(question, answer, context)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"\nReflection Results:")
    print(f"  Is Satisfactory: {reflection.is_satisfactory}")
    print(f"  Confidence: {reflection.confidence:.2f}")
    print(f"  Should Retry: {reflection.should_retry}")
    print(f"  Issues: {reflection.issues}")
    print(f"  Suggestions: {reflection.suggestions}")
    
    print("\n✓ Self-reflection test completed\n")
    return reflection


def test_full_pipeline():
    """Test the full ReasoningAgent pipeline."""
    print("=" * 60)
    print("Test 3: Full Pipeline")
    print("=" * 60)
    
    agent = ReasoningAgent()
    
    question = "Compare the approaches of RAG and traditional fine-tuning for knowledge-intensive tasks."
    context = [
        "RAG retrieves relevant documents and uses them to augment generation.",
        "Fine-tuning updates model parameters on task-specific data.",
        "RAG is more parameter-efficient as it doesn't require model updates.",
        "Fine-tuning can achieve better performance on specific domains but requires more data.",
    ]
    
    input_data = AgentInput(
        question=question,
        context=context,
    )
    
    output = agent.run(input_data)
    
    print(f"Question: {question}")
    print(f"\nAnswer: {output.answer}")
    print(f"\nConfidence: {output.confidence:.2f}")
    print(f"Should Retry: {output.should_retry}")
    print(f"\nEvidence used: {len(output.evidence)} passages")
    
    # Print metadata
    metadata = output.metadata
    if "decomposition" in metadata:
        decomp = metadata["decomposition"]
        print(f"\nDecomposition:")
        print(f"  Sub-questions: {len(decomp.get('sub_questions', []))}")
        print(f"  Estimated hops: {decomp.get('estimated_hops', 'N/A')}")
    
    print("\n✓ Full pipeline test completed\n")
    return output


def test_retry_mechanism():
    """Test the retry mechanism with self-reflection."""
    print("=" * 60)
    print("Test 4: Retry Mechanism")
    print("=" * 60)
    
    agent = ReasoningAgent(
        confidence_threshold=0.8,  # High threshold to trigger retry
        max_retries=2,
    )
    
    question = "Explain the impact of attention mechanisms on NLP."
    context = [
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "The transformer architecture is based entirely on attention mechanisms.",
    ]
    
    input_data = AgentInput(
        question=question,
        context=context,
    )
    
    # First attempt
    output = agent.run(input_data)
    print(f"First attempt confidence: {output.confidence:.2f}")
    print(f"Should retry: {output.should_retry}")
    
    # If retry is needed, demonstrate retry
    if output.should_retry:
        from rag_qa.agents.reasoning_agent import ReflectionResult
        
        # Create a mock reflection that suggests retry
        reflection = ReflectionResult(
            is_satisfactory=False,
            confidence=output.confidence,
            issues=["Answer lacks depth", "Missing specific examples"],
            suggestions=["Include more technical details", "Add concrete examples"],
            should_retry=True,
            retry_strategy="Add more detailed explanation",
        )
        
        retry_output = agent.retry_with_strategy(input_data, output, reflection)
        print(f"\nRetry attempt confidence: {retry_output.confidence:.2f}")
        print(f"Retry count: {retry_output.metadata.get('retry_count', 0)}")
    
    print("\n✓ Retry mechanism test completed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Reasoning Agent Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Run tests
        test_basic_decomposition()
        test_self_reflection()
        test_full_pipeline()
        test_retry_mechanism()
        
        print("=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
