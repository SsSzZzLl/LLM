import json
import urllib.request
import os
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

SQUAD_OUT = DATA_DIR / "eval_squad_moderate.jsonl"
HOTPOT_OUT = DATA_DIR / "eval_hotpot_complex.jsonl"

def download_and_format_squad(num_samples=50):
    print(f"Downloading SQuAD 2.0 (for MODERATE queries)...")
    req = urllib.request.Request(SQUAD_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    samples = []
    # Parse SQuAD
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if not qa['is_impossible'] and qa['answers']:
                    samples.append({
                        "question": qa['question'],
                        "gold_answer": qa['answers'][0]['text']
                    })
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
            
    with open(SQUAD_OUT, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
            
    print(f"✅ Saved {len(samples)} SQuAD queries to {SQUAD_OUT.name}")

def download_and_format_hotpot(num_samples=50):
    print(f"Downloading HotpotQA (for COMPLEX multi-hop queries)...")
    req = urllib.request.Request(HOTPOT_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    
    samples = []
    # Parse HotpotQA
    for item in data:
        samples.append({
            "question": item['question'],
            "gold_answer": item['answer']
        })
        if len(samples) >= num_samples:
            break
            
    with open(HOTPOT_OUT, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
            
    print(f"✅ Saved {len(samples)} HotpotQA queries to {HOTPOT_OUT.name}")

if __name__ == "__main__":
    try:
        download_and_format_squad(50)
    except Exception as e:
        print(f"Failed to download SQuAD: {e}")
        
    try:
        download_and_format_hotpot(50)
    except Exception as e:
        print(f"Failed to download HotpotQA: {e}")
    
    print("\nDatasets are ready! You can test them using:")
    print(f"python -m rag_qa.cli eval data/{SQUAD_OUT.name}")
    print(f"python -m rag_qa.cli eval data/{HOTPOT_OUT.name}")
