"""测试 OpenRouter 上可用的模型"""
import requests

with open('.env', 'r') as f:
    lines = f.readlines()
    
api_key = None
base_url = None

for line in lines:
    line = line.strip()
    if line.startswith('OPENAI_API_KEY='):
        api_key = line.split('=', 1)[1]
    elif line.startswith('OPENAI_BASE_URL='):
        base_url = line.split('=', 1)[1]

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://github.com/llm-group/rag-qa",
    "X-Title": "CDS547 RAG QA Project",
}

# 尝试不同的模型
models_to_try = [
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-7b-instruct",
]

print("Testing available models on OpenRouter:\n")

for model in models_to_try:
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello' and nothing else."}
        ],
        "max_tokens": 10,
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✓ {model}: {content[:50]}")
        else:
            error = response.json().get('error', {}).get('message', 'Unknown error')
            print(f"✗ {model}: {error}")
    except Exception as e:
        print(f"✗ {model}: {e}")
