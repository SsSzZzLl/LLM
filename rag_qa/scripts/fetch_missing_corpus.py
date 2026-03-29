import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_DIR.mkdir(exist_ok=True, parents=True)

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

def fetch_contexts():
    print("⏳ Downloading SQuAD contexts...")
    try:
        req = urllib.request.Request(SQUAD_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            squad_data = json.loads(response.read().decode())
        
        squad_txt = CORPUS_DIR / "squad_paragraphs.txt"
        with open(squad_txt, 'w', encoding='utf-8') as f:
            count = 0
            for article in squad_data['data']:
                for paragraph in article['paragraphs']:
                    f.write(paragraph['context'].replace('\n', ' ') + '\n\n')
                    count += 1
                    if count >= 100: # Just top 100 paragraphs to cover the 50 questions
                        break
                if count >= 100: break
        print(f"✅ Saved 100 SQuAD Wikipedia paragraphs to {squad_txt.name}")
    except Exception as e:
        print(f"❌ Failed to fetch SQuAD corpus: {e}")

    print("⏳ Processing HotpotQA contexts...")
    try:
        hotpot_local = DATA_DIR / "hotpot_dev_distractor_v1.json"
        if hotpot_local.exists():
            print(f"📖 Found local file at {hotpot_local.name}, parsing directly...")
            with open(hotpot_local, 'r', encoding='utf-8') as f:
                hotpot_data = json.load(f)
        else:
            print(f"🌐 No local file {hotpot_local.name} found. Skipping network download... \n⚠️ 请用浏览器下载该文件并放入 data/ 文件夹中。")
            return
        
        
        hotpot_txt = CORPUS_DIR / "hotpot_paragraphs.txt"
        with open(hotpot_txt, 'w', encoding='utf-8') as f:
            # The first 50 items correspond to our eval jsonl
            for item in hotpot_data[:50]:
                for title, sentences in item['context']:
                    f.write(f"Title: {title}\n" + " ".join(sentences) + "\n\n")
        print(f"✅ Saved corresponding HotpotQA Wikipedia contexts to {hotpot_txt.name}")
    except Exception as e:
        print(f"❌ Failed to fetch HotpotQA corpus: {e}")

if __name__ == "__main__":
    fetch_contexts()
