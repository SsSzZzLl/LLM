"""
Route Agent 真实 API 测试
使用实际的 LLM 进行问题复杂度分类
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from rag_qa.agents.route_agent import RouteAgent, QuestionComplexity


def test_real_classification():
    """使用真实 LLM 进行分类测试"""
    print("=" * 70)
    print("Route Agent - 真实 LLM 分类测试")
    print("=" * 70)
    print(f"\n使用模型: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print("-" * 70)
    
    agent = RouteAgent()
    
    # 测试问题集
    test_questions = [
        # SIMPLE - 简单事实
        "What is 2+2?",
        "Who wrote Romeo and Juliet?",
        
        # MODERATE - 需要检索
        "What does RAG condition the generator on?",
        "What architecture did Vaswani et al. propose in 2017?",
        
        # COMPLEX - 需要推理
        "Compare the RAG approach by Lewis et al. with the transformer architecture by Vaswani et al.",
        "What are the trade-offs between using dense retrieval versus BM25 in multi-hop question answering?",
    ]
    
    results = []
    for question in test_questions:
        print(f"\n❓ 问题: {question}")
        
        try:
            decision, config = agent.route(question)
            
            print(f"   ├─ 复杂度: {decision.complexity.value.upper()}")
            print(f"   ├─ 置信度: {decision.confidence:.1%}")
            print(f"   ├─ 策略: {config['name']}")
            print(f"   ├─ 使用检索: {'✓' if config['use_retrieval'] else '✗'}")
            print(f"   ├─ 多跳推理: {'✓' if config['use_multi_hop'] else '✗'}")
            print(f"   └─ LLM 推理: {decision.reasoning}")
            
            results.append({
                'question': question,
                'complexity': decision.complexity.value,
                'confidence': decision.confidence,
                'strategy': config['name'],
                'use_retrieval': config['use_retrieval'],
                'use_multi_hop': config['use_multi_hop'],
            })
            
        except Exception as e:
            print(f"   └─ ❌ 错误: {e}")
    
    # 统计结果
    print("\n" + "=" * 70)
    print("分类统计")
    print("=" * 70)
    
    simple_count = sum(1 for r in results if r['complexity'] == 'simple')
    moderate_count = sum(1 for r in results if r['complexity'] == 'moderate')
    complex_count = sum(1 for r in results if r['complexity'] == 'complex')
    
    print(f"\nSIMPLE (直接生成):   {simple_count} 个")
    print(f"MODERATE (单跳检索): {moderate_count} 个")
    print(f"COMPLEX (多跳推理):  {complex_count} 个")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"\n平均置信度: {avg_confidence:.1%}")
    
    return results


def test_single_question(question: str):
    """测试单个问题"""
    print("\n" + "=" * 70)
    print("单问题测试")
    print("=" * 70)
    
    agent = RouteAgent()
    
    print(f"\n问题: {question}")
    print("-" * 70)
    
    try:
        decision, config = agent.route(question)
        
        print(f"\n📊 分类结果:")
        print(f"   复杂度级别: {decision.complexity.value.upper()}")
        print(f"   置信度: {decision.confidence:.1%}")
        print(f"   推荐策略: {decision.recommended_strategy}")
        
        print(f"\n🔧 路由配置:")
        print(f"   策略名称: {config['name']}")
        print(f"   策略描述: {config['description']}")
        print(f"   启用检索: {config['use_retrieval']}")
        print(f"   多跳推理: {config['use_multi_hop']}")
        
        print(f"\n💭 LLM 推理过程:")
        print(f"   {decision.reasoning}")
        
        print(f"\n📋 建议执行流程:")
        if not config['use_retrieval']:
            print("   → Direct Generation Agent")
        elif not config['use_multi_hop']:
            print("   → Retrieval Agent → Synthesis Agent")
        else:
            print("   → Reasoning Agent → Retrieval Agent (迭代) → Synthesis Agent")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("❌ 错误: 未找到 OPENAI_API_KEY")
        print("请在 .env 文件中配置 API Key")
        exit(1)
    
    print(f"✓ API Key 已配置: {api_key[:20]}...")
    
    # 运行批量测试
    results = test_real_classification()
    
    # 测试自定义问题（可选）
    print("\n" + "=" * 70)
    custom = input("\n输入自定义问题测试（直接回车跳过）: ").strip()
    if custom:
        test_single_question(custom)
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
