import json
with open('data/traces/all_traces_multi-agent_orchestrator_rag_routing.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('data/traces/stats.log', 'w', encoding='utf-8') as f:
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
        if 'Fallback parse' in str(d.get('trace', '')):
            fallback += 1
        
        telemetry = d.get('telemetry')
        if not telemetry:
            telemetry = {}
        if telemetry.get('total_completion_tokens', 0) >= 2000:
            high_tokens += 1
            
    f.write(f'Total questions evaluated: {total}\n')
    f.write(f'Route Misclassifications: {misclass} ({(misclass/total)*100:.1f}%)\n')
    f.write(f'Rough Contains/EM Rate: {em} ({(em/total)*100:.1f}%)\n')
    f.write(f'Fallback parsing failures: {fallback} ({(fallback/total)*100:.1f}%)\n')
    f.write(f'High token output (>2000): {high_tokens} ({(high_tokens/total)*100:.1f}%)\n\n')
    
    for i, d in enumerate(data[:3]):
        q = d.get('question')
        c = d.get('complexity')
        g = d.get('gold')
        a = str(d.get('exact_answer'))[:150].replace('\n', ' ')
        
        tele = d.get('telemetry')
        if not tele:
            tele = {}
        pin = tele.get('total_prompt_tokens', 0)
        pout = tele.get('total_completion_tokens', 0)
        
        f.write(f'\nQ{i+1}: {q}\n')
        f.write(f'Predicted Complexity: {c}\n')
        f.write(f'Gold: {g}\n')
        f.write(f'Agent Answer: {a}\n')
        f.write(f'Tokens: In={pin}, Out={pout}\n')
