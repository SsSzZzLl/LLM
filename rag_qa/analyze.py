import json

def analyze():
    with open('data/traces/all_traces_multi-agent_orchestrator_rag_routing.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    misclass = sum(1 for d in data if d.get('complexity') != 'complex')
    
    em = 0
    fallback = 0
    high_tokens = 0
    
    for d in data:
        gold = d.get('gold')
        ans = str(d.get('exact_answer', '')).lower().strip()
        if gold and str(gold).lower().strip() in ans:
            em += 1
            
        trace = str(d.get('trace', ''))
        if 'Fallback parse' in trace:
            fallback += 1
            
        telemetry = d.get('telemetry', {})
        if telemetry:
            if telemetry.get('total_completion_tokens', 0) >= 2000:
                high_tokens += 1

    print(f'Total questions evaluated: {total}')
    print(f'Route Misclassifications (non-COMPLEX): {misclass} ({(misclass/total)*100:.1f}%)')
    print(f'Rough Contains/EM Rate: {em} ({(em/total)*100:.1f}%)')
    print(f'Fallback parsing failures (JSON error): {fallback} ({(fallback/total)*100:.1f}%)')
    print(f'High token output (>2000 tokens per question): {high_tokens} ({(high_tokens/total)*100:.1f}%)')

    print("\n--- Failure Examples (Q 1-3) ---")
    for i, d in enumerate(data[:3]):
        print(f"\nQ{i+1}: {d.get('question')}")
        print(f"Predicted Complexity: {d.get('complexity')}")
        print(f"Gold: {d.get('gold')}")
        print(f"Agent Final Answer: {str(d.get('exact_answer'))[:150]}")
        tele = d.get('telemetry', {})
        print(f"Tokens: Prompt={tele.get('total_prompt_tokens')}, Completion={tele.get('total_completion_tokens')}")

if __name__ == '__main__':
    analyze()
