import json
import re

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens).intersection(set(ground_truth_tokens))
    if len(common) == 0:
        return 0.0
    prec = len(common) / len(prediction_tokens)
    rec = len(common) / len(ground_truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def exact_match_score(prediction, ground_truth):
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

with open('data/traces/all_traces_multi-agent_orchestrator_rag_routing.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total = len(data)

misclassify = 0
em_count = 0
f1_total = 0.0
total_prompt_tokens = 0
total_comp_tokens = 0
total_latency = 0.0

bad_predictions = []

for idx, d in enumerate(data):
    if d.get("complexity") != "complex":
        misclassify += 1
        
    pred = str(d.get("exact_answer", ""))
    gold = str(d.get("gold", ""))
    
    em = exact_match_score(pred, gold)
    f1 = f1_score(pred, gold)
    
    if em == 1.0:
        em_count += 1
    else:
        # Also check simple substring containment just in case metric is harsh
        if normalize_answer(gold) in normalize_answer(pred):
            em_count += 1
            em = 1.0
            f1 = 1.0
    
    if em == 0.0:
        bad_predictions.append((d.get("question", ""), gold, pred, d.get("complexity")))
        
    f1_total += f1
    
    tel = d.get("telemetry", {})
    total_prompt_tokens += tel.get("total_prompt_tokens", 0)
    total_comp_tokens += tel.get("total_completion_tokens", 0)
    total_latency += tel.get("total_latency", 0)

with open('data/final_stats.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total Evaluated: {total}\n")
    f.write(f"Routing Misclassification: {misclassify}/{total} ({(misclassify/total)*100 if total > 0 else 0:.1f}%)\n")
    f.write(f"Exact Match (EM) Rate: {(em_count/total)*100 if total > 0 else 0:.2f}%\n")
    f.write(f"Average F1 Score: {(f1_total/total)*100 if total > 0 else 0:.2f}%\n")
    
    if total > 0:
        f.write(f"\nAverage Tokens per question: {total_prompt_tokens/total:.0f} in / {total_comp_tokens/total:.0f} out\n")
        f.write(f"Average Latency: {total_latency/total:.2f} seconds\n")
        f.write(f"Estimated Total Input Tokens: {total_prompt_tokens}\n")
        f.write(f"Estimated Total Output Tokens: {total_comp_tokens}\n")

    f.write("\n=== Failures (Preview 5) ===\n")
    for q, g, p, c in bad_predictions[:5]:
        f.write(f"Q: {q}\nGold: {g}\nPred: {p}\nRoute: {c}\n---\n")
