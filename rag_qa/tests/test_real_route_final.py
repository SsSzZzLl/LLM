"""
Route Agent 真实 API 测试 - 使用 OpenRouter
"""
import os
import sys

# 直接从文件读取配置
with open('.env', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
for line in lines:
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.split('=', 1)
        os.environ[key] = value

from rag_qa.agents.route_agent import RouteAgent, QuestionComplexity


def test_real_classification():
    """使用真实 LLM 进行分类测试"""
    print("=" * 70)
    print("Route Agent - 真实 LLM 分类测试 (OpenRouter)")
    print("=" * 70)
    print(f"\n使用模型: {os.getenv('OPENAI_MODEL')}")
    print(f"API: {os.getenv('OPENAI_BASE_URL')}")
    print("-" * 70)
    
    agent = RouteAgent()
    
    # 测试问题集
    test_questions = [
        # SIMPLE - 简单事实
        ("What is 2+2?", QuestionComplexity.SIMPLE),
        ("Who wrote Romeo and Juliet?", QuestionComplexity.SIMPLE),
        
        # MODERATE - 需要检索
        ("What does RAG condition the generator on?", QuestionComplexity.MODERATE),
        ("What architecture did Vaswani et al. propose in 2017?", QuestionComplexity.MODERATE),
        
        # COMPLEX - 需要推理
        ("Compare the RAG approach by Lewis et al. with the transformer architecture by Vaswani et al.", QuestionComplexity.COMPLEX),
        ("What are the trade-offs between using dense retrieval versus BM25 in multi-hop question answering?", QuestionComplexity.COMPLEX),
    ]
    
    results = []
    correct = 0
    
    for question, expected in test_questions:
        print(f"\n问题: {question}")
        print(f"   期望复杂度: {expected.value}")
        
        try:
            decision = agent.classify(question)
            strategy = agent.ROUTING_STRATEGIES[decision.complexity]
            
            is_correct = decision.complexity == expected
            if is_correct:
                correct += 1
                status = "OK"
            else:
                status = "XX"
            
            print(f"   [{status}] 预测: {decision.complexity.value.upper()}")
            print(f"   置信度: {decision.confidence:.1%}")
            print(f"   策略: {strategy['name']}")
            print(f"   检索: {'Y' if strategy['use_retrieval'] else 'N'} | 多跳: {'Y' if strategy['use_multi_hop'] else 'N'}")
            print(f"   推理: {decision.reasoning[:60]}...")
            
            results.append({
                'question': question,
                'expected': expected.value,
                'predicted': decision.complexity.value,
                'correct': is_correct,
                'confidence': decision.confidence,
            })
            
        except Exception as e:
            import traceback
            print(f"   [ERR] {e}")
            traceback.print_exc()
    
    # 统计结果
    print("\n" + "=" * 70)
    print("分类统计")
    print("=" * 70)
    
    accuracy = correct / len(test_questions) if test_questions else 0
    print(f"\n准确率: {accuracy:.1%} ({correct}/{len(test_questions)})")
    
    simple_count = sum(1 for r in results if r['predicted'] == 'simple')
    moderate_count = sum(1 for r in results if r['predicted'] == 'moderate')
    complex_count = sum(1 for r in results if r['predicted'] == 'complex')
    
    print(f"\n分类分布:")
    print(f"  SIMPLE:   {simple_count} 个")
    print(f"  MODERATE: {moderate_count} 个")
    print(f"  COMPLEX:  {complex_count} 个")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"\n平均置信度: {avg_confidence:.1%}")
    
    return results


if __name__ == "__main__":
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("错误: 未找到 API Key")
        sys.exit(1)
    
    print(f"API Key: {api_key[:20]}...")
    
    # 运行测试
    results = test_real_classification()
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
