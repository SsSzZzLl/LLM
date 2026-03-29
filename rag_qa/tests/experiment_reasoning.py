"""
Experiment Script for Reasoning Agent

This script runs comprehensive experiments to evaluate:
1. Question decomposition quality
2. Self-reflection accuracy
3. Multi-hop reasoning performance
4. Retry mechanism effectiveness

Author: 叶子冉
Date: 2026-03-29
"""

import sys
import os
import json
import time
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_qa.agents import ReasoningAgent, AgentInput
from rag_qa.agents.reasoning_agent import ReasoningEvaluator


class ReasoningExperiment:
    """Comprehensive experiment suite for Reasoning Agent."""
    
    def __init__(self, output_dir="../experiments"):
        self.agent = ReasoningAgent(
            confidence_threshold=0.7,
            max_retries=2,
        )
        self.evaluator = ReasoningEvaluator()
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_all_experiments(self):
        """Run all experiments and save results."""
        print("=" * 70)
        print("Reasoning Agent - Comprehensive Experiment Suite")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Experiment 1: Question Decomposition
        self.experiment_decomposition()
        
        # Experiment 2: Self-Reflection
        self.experiment_self_reflection()
        
        # Experiment 3: Multi-hop QA
        self.experiment_multi_hop_qa()
        
        # Experiment 4: Retry Mechanism
        self.experiment_retry_mechanism()
        
        # Save results
        self.save_results()
        
        print()
        print("=" * 70)
        print("All experiments completed!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
    
    def experiment_decomposition(self):
        """Experiment 1: Evaluate question decomposition quality."""
        print("\n" + "=" * 70)
        print("Experiment 1: Question Decomposition Quality")
        print("=" * 70)
        
        test_cases = [
            {
                "question": "What is the capital of France?",
                "type": "simple",
                "expected_hops": 1,
            },
            {
                "question": "What is the relationship between RAG and transformer architecture?",
                "type": "moderate",
                "expected_hops": 2,
            },
            {
                "question": "Compare the approaches of RAG and traditional fine-tuning for knowledge-intensive tasks, and explain which is more suitable for small teams.",
                "type": "complex",
                "expected_hops": 3,
            },
            {
                "question": "How does the attention mechanism in transformers enable RAG systems to better retrieve relevant information?",
                "type": "complex",
                "expected_hops": 3,
            },
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {case['type'].upper()}")
            print(f"Question: {case['question']}")
            
            context = [
                "RAG combines retrieval and generation.",
                "Transformers use attention mechanisms.",
                "Fine-tuning updates model parameters.",
            ]
            
            start_time = time.time()
            decomposition = self.agent.decompose_question(case["question"], context)
            elapsed_time = time.time() - start_time
            
            result = {
                "test_id": i,
                "question": case["question"],
                "type": case["type"],
                "expected_hops": case["expected_hops"],
                "actual_hops": decomposition.estimated_hops,
                "num_sub_questions": len(decomposition.sub_questions),
                "reasoning_plan": decomposition.reasoning_plan,
                "elapsed_time": elapsed_time,
            }
            results.append(result)
            
            print(f"  Expected hops: {case['expected_hops']}")
            print(f"  Actual hops: {decomposition.estimated_hops}")
            print(f"  Sub-questions: {len(decomposition.sub_questions)}")
            print(f"  Time: {elapsed_time:.3f}s")
        
        # Summary
        avg_sub_qs = sum(r["num_sub_questions"] for r in results) / len(results)
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        
        print(f"\n--- Summary ---")
        print(f"Total test cases: {len(results)}")
        print(f"Average sub-questions: {avg_sub_qs:.2f}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        self.results.append({
            "experiment": "decomposition",
            "results": results,
            "summary": {
                "avg_sub_questions": avg_sub_qs,
                "avg_time": avg_time,
            }
        })
    
    def experiment_self_reflection(self):
        """Experiment 2: Evaluate self-reflection mechanism."""
        print("\n" + "=" * 70)
        print("Experiment 2: Self-Reflection Mechanism")
        print("=" * 70)
        
        test_cases = [
            {
                "question": "What is 2+2?",
                "answer": "2+2 equals 4.",
                "context": ["Basic arithmetic: 2+2=4"],
                "expected_satisfactory": True,
            },
            {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "context": ["Paris is the capital of France."],
                "expected_satisfactory": True,
            },
            {
                "question": "Explain quantum computing.",
                "answer": "Quantum computing uses qubits.",
                "context": ["Quantum computing is a type of computation."],
                "expected_satisfactory": False,  # Too brief
            },
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}")
            print(f"Question: {case['question']}")
            print(f"Answer: {case['answer']}")
            
            reflection = self.agent.reflect(
                case["question"],
                case["answer"],
                case["context"],
            )
            
            result = {
                "test_id": i,
                "question": case["question"],
                "answer": case["answer"],
                "expected_satisfactory": case["expected_satisfactory"],
                "actual_satisfactory": reflection.is_satisfactory,
                "confidence": reflection.confidence,
                "should_retry": reflection.should_retry,
                "issues": reflection.issues,
                "suggestions": reflection.suggestions,
            }
            results.append(result)
            
            print(f"  Expected satisfactory: {case['expected_satisfactory']}")
            print(f"  Actual satisfactory: {reflection.is_satisfactory}")
            print(f"  Confidence: {reflection.confidence:.2f}")
            print(f"  Should retry: {reflection.should_retry}")
        
        # Calculate accuracy
        correct = sum(
            1 for r in results
            if r["expected_satisfactory"] == r["actual_satisfactory"]
        )
        accuracy = correct / len(results)
        
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        print(f"\n--- Summary ---")
        print(f"Total test cases: {len(results)}")
        print(f"Reflection accuracy: {accuracy:.2%}")
        print(f"Average confidence: {avg_confidence:.2f}")
        
        self.results.append({
            "experiment": "self_reflection",
            "results": results,
            "summary": {
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
            }
        })
    
    def experiment_multi_hop_qa(self):
        """Experiment 3: Multi-hop QA performance."""
        print("\n" + "=" * 70)
        print("Experiment 3: Multi-hop QA Performance")
        print("=" * 70)
        
        test_cases = [
            {
                "question": "What is the relationship between transformers and BERT?",
                "context": [
                    "Transformers were introduced in 2017 by Vaswani et al.",
                    "BERT is a transformer-based model developed by Google.",
                    "BERT stands for Bidirectional Encoder Representations from Transformers.",
                ],
                "requires_hops": 2,
            },
            {
                "question": "How does RAG improve upon traditional language models?",
                "context": [
                    "Traditional language models rely only on parametric knowledge.",
                    "RAG retrieves external documents to augment generation.",
                    "RAG was introduced by Lewis et al. in 2020.",
                ],
                "requires_hops": 2,
            },
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}")
            print(f"Question: {case['question']}")
            print(f"Required hops: {case['requires_hops']}")
            
            input_data = AgentInput(
                question=case["question"],
                context=case["context"],
            )
            
            start_time = time.time()
            output = self.agent.run(input_data)
            elapsed_time = time.time() - start_time
            
            result = {
                "test_id": i,
                "question": case["question"],
                "requires_hops": case["requires_hops"],
                "answer": output.answer[:200] + "..." if len(output.answer) > 200 else output.answer,
                "confidence": output.confidence,
                "should_retry": output.should_retry,
                "num_evidence": len(output.evidence),
                "elapsed_time": elapsed_time,
            }
            results.append(result)
            
            print(f"  Confidence: {output.confidence:.2f}")
            print(f"  Should retry: {output.should_retry}")
            print(f"  Evidence count: {len(output.evidence)}")
            print(f"  Time: {elapsed_time:.3f}s")
            print(f"  Answer preview: {result['answer']}")
        
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        
        print(f"\n--- Summary ---")
        print(f"Total test cases: {len(results)}")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        self.results.append({
            "experiment": "multi_hop_qa",
            "results": results,
            "summary": {
                "avg_confidence": avg_confidence,
                "avg_time": avg_time,
            }
        })
    
    def experiment_retry_mechanism(self):
        """Experiment 4: Retry mechanism effectiveness."""
        print("\n" + "=" * 70)
        print("Experiment 4: Retry Mechanism Effectiveness")
        print("=" * 70)
        
        from rag_qa.agents.reasoning_agent import ReflectionResult
        
        question = "Explain the attention mechanism in detail."
        context = [
            "Attention mechanisms allow models to focus on relevant parts.",
            "Self-attention computes attention over the same sequence.",
        ]
        
        input_data = AgentInput(
            question=question,
            context=context,
        )
        
        print(f"\nQuestion: {question}")
        print("Running with retry simulation...")
        
        # First attempt (low confidence)
        output1 = self.agent.run(input_data)
        print(f"\nAttempt 1:")
        print(f"  Confidence: {output1.confidence:.2f}")
        print(f"  Should retry: {output1.should_retry}")
        
        # Simulate retry
        if output1.should_retry:
            reflection = ReflectionResult(
                is_satisfactory=False,
                confidence=output1.confidence,
                issues=["Insufficient detail", "Missing examples"],
                suggestions=["Add technical details", "Include concrete examples"],
                should_retry=True,
                retry_strategy="Enhance with more details",
            )
            
            output2 = self.agent.retry_with_strategy(input_data, output1, reflection)
            print(f"\nAttempt 2 (Retry):")
            print(f"  Confidence: {output2.confidence:.2f}")
            print(f"  Retry count: {output2.metadata.get('retry_count', 0)}")
        
        result = {
            "question": question,
            "attempt_1_confidence": output1.confidence,
            "attempt_1_should_retry": output1.should_retry,
            "attempt_2_confidence": output2.confidence if output1.should_retry else None,
            "retry_count": output2.metadata.get('retry_count', 0) if output1.should_retry else 0,
        }
        
        print(f"\n--- Summary ---")
        print(f"Retry mechanism successfully triggered: {output1.should_retry}")
        
        self.results.append({
            "experiment": "retry_mechanism",
            "results": [result],
            "summary": {
                "retry_triggered": output1.should_retry,
            }
        })
    
    def save_results(self):
        """Save all experiment results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"reasoning_experiment_results_{timestamp}.json"
        )
        
        experiment_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "agent_config": {
                    "confidence_threshold": self.agent.confidence_threshold,
                    "max_retries": self.agent.max_retries,
                },
            },
            "experiments": self.results,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point."""
    experiment = ReasoningExperiment()
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()
