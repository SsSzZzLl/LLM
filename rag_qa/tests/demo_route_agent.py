"""
Route Agent Demo - 模拟演示动态路由效果
由于 API 限制，使用模拟数据展示 Route Agent 的工作流程
"""
from rag_qa.agents.route_agent import (
    RouteAgent, 
    RouteDecision, 
    RouteEvaluator,
    QuestionComplexity, 
    create_test_dataset
)


def mock_classify(agent, question):
    """模拟分类结果（实际使用时调用 agent.classify）"""
    # 基于关键词的简单模拟逻辑
    q_lower = question.lower()
    
    # SIMPLE 问题特征
    simple_keywords = ['what is', 'define', 'who wrote', 'capital of', '2+2', 'meaning of']
    # COMPLEX 问题特征  
    complex_keywords = ['compare', 'contrast', 'trade-offs', 'implications', 'how does', 'relate to', 'multi-hop']
    
    if any(kw in q_lower for kw in simple_keywords):
        return RouteDecision(
            complexity=QuestionComplexity.SIMPLE,
            confidence=0.92,
            reasoning="Question asks for basic factual information that can be answered directly without document retrieval.",
            recommended_strategy="direct_generation"
        )
    elif any(kw in q_lower for kw in complex_keywords):
        return RouteDecision(
            complexity=QuestionComplexity.COMPLEX,
            confidence=0.88,
            reasoning="Question requires comparing multiple concepts and synthesizing information from different sources.",
            recommended_strategy="multi_hop_reasoning"
        )
    else:
        return RouteDecision(
            complexity=QuestionComplexity.MODERATE,
            confidence=0.85,
            reasoning="Question requires specific information from documents but can be answered with single-hop retrieval.",
            recommended_strategy="single_hop_rag"
        )


def demo_classification():
    """演示问题分类效果"""
    print("=" * 70)
    print("Route Agent - 问题复杂度分类演示")
    print("=" * 70)
    
    agent = RouteAgent()
    
    test_questions = [
        # SIMPLE 问题
        "What is 2+2?",
        "Define machine learning.",
        "Who wrote Romeo and Juliet?",
        "What is the capital of France?",
        
        # MODERATE 问题
        "What does RAG condition the generator on?",
        "Who introduced RAG for knowledge-intensive tasks?",
        "What architecture did Vaswani et al. propose in 2017?",
        "What is hallucination in the context of LLM evaluation?",
        
        # COMPLEX 问题
        "Compare the RAG approach by Lewis et al. with the transformer architecture by Vaswani et al.",
        "How does the hallucination problem in LLMs relate to the retrieval mechanism in RAG systems?",
        "What are the trade-offs between using dense retrieval versus BM25 in multi-hop QA?",
        "Explain how self-reflection mechanisms in agent systems could improve retrieval quality.",
    ]
    
    for question in test_questions:
        decision = mock_classify(agent, question)
        config = agent.ROUTING_STRATEGIES[decision.complexity]
        
        print(f"\n❓ 问题: {question}")
        print(f"   ├─ 复杂度: {decision.complexity.value.upper()}")
        print(f"   ├─ 置信度: {decision.confidence:.0%}")
        print(f"   ├─ 策略: {config['name']}")
        print(f"   ├─ 使用检索: {config['use_retrieval']}")
        print(f"   ├─ 多跳推理: {config['use_multi_hop']}")
        print(f"   └─ 推理: {decision.reasoning}")


def demo_evaluation():
    """演示评估效果"""
    print("\n" + "=" * 70)
    print("Route Agent - 路由准确率评估")
    print("=" * 70)
    
    agent = RouteAgent()
    test_data = create_test_dataset()
    evaluator = RouteEvaluator()
    
    print(f"\n测试集: {len(test_data)} 个问题")
    print("-" * 70)
    
    correct = 0
    for question, ground_truth in test_data:
        decision = mock_classify(agent, question)
        is_correct = decision.complexity == ground_truth
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} [{decision.complexity.value:8}] {question[:50]}...")
        evaluator.add_prediction(question, decision, ground_truth)
    
    # 计算指标
    accuracy = correct / len(test_data)
    
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"\n整体准确率: {accuracy:.1%} ({correct}/{len(test_data)})")
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    print(f"{'预测\\真实':12} {'simple':>8} {'moderate':>8} {'complex':>8}")
    print("-" * 40)
    
    cm = {'simple': {}, 'moderate': {}, 'complex': {}}
    for pred in ['simple', 'moderate', 'complex']:
        for true in ['simple', 'moderate', 'complex']:
            count = sum(1 for q, d, g in zip(evaluator.questions, evaluator.predictions, evaluator.ground_truth)
                       if d.complexity.value == pred and g.value == true)
            cm[pred][true] = count
    
    for pred_cls in ['simple', 'moderate', 'complex']:
        row = f"{pred_cls:12}"
        for true_cls in ['simple', 'moderate', 'complex']:
            row += f"{cm[pred_cls][true_cls]:>8}"
        print(row)


def demo_routing_flow():
    """演示完整路由流程"""
    print("\n" + "=" * 70)
    print("Route Agent - 完整路由流程演示")
    print("=" * 70)
    
    agent = RouteAgent()
    
    examples = [
        ("What is the capital of Japan?", "简单事实问题"),
        ("What does RAG condition the generator on?", "课程相关问题"),
        ("Compare dense retrieval and BM25 for multi-hop QA.", "需要比较分析"),
    ]
    
    for question, desc in examples:
        print(f"\n📥 输入问题 ({desc}):")
        print(f"   \"{question}\"")
        
        decision, config = agent.route(question)
        # 使用模拟结果覆盖
        decision = mock_classify(agent, question)
        config = agent.ROUTING_STRATEGIES[decision.complexity]
        
        print(f"\n   ↓ Route Agent 分析")
        print(f"   ├─ 复杂度识别: {decision.complexity.value}")
        print(f"   ├─ 置信度: {decision.confidence:.0%}")
        print(f"   └─ 推理: {decision.reasoning[:60]}...")
        
        print(f"\n   ↓ 路由决策")
        print(f"   ├─ 选择策略: {config['name']}")
        print(f"   ├─ 策略描述: {config['description']}")
        print(f"   ├─ 启用检索: {'✓' if config['use_retrieval'] else '✗'}")
        print(f"   └─ 多跳推理: {'✓' if config['use_multi_hop'] else '✗'}")
        
        print(f"\n   ↓ 执行流程")
        if not config['use_retrieval']:
            print("   └─ → Direct Generation Agent")
        elif not config['use_multi_hop']:
            print("   └─ → Retrieval Agent → Synthesis Agent")
        else:
            print("   └─ → Reasoning Agent → Retrieval Agent (迭代) → Synthesis Agent")


if __name__ == "__main__":
    demo_classification()
    demo_evaluation()
    demo_routing_flow()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n说明: 本演示使用模拟分类器展示 Route Agent 的工作流程。")
    print("实际使用时，将调用 LLM (gpt-4o-mini) 进行真实分类。")
    print("运行: python -m rag_qa route-demo")
