"""直接测试 OpenRouter API - 使用正确的 API Key"""
import requests
import json

# 直接从文件读取，确保使用正确的 key
with open('.env', 'r') as f:
    lines = f.readlines()
    
api_key = None
base_url = None
model = None

for line in lines:
    line = line.strip()
    if line.startswith('OPENAI_API_KEY='):
        api_key = line.split('=', 1)[1]
    elif line.startswith('OPENAI_BASE_URL='):
        base_url = line.split('=', 1)[1]
    elif line.startswith('OPENAI_MODEL='):
        model = line.split('=', 1)[1]

print(f"API Key: {api_key[:25]}..." if api_key else "API Key: NOT FOUND")
print(f"Base URL: {base_url}")
print(f"Model: {model}")
print()

if not api_key:
    print("Error: API Key not found")
    exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/llm-group/rag-qa",
    "X-Title": "CDS547 RAG QA Project",
}

data = {
    "model": model or "openai/gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer with a JSON object: {\"answer\": number, \"explanation\": string}"}
    ],
    "temperature": 0.1,
    "max_tokens": 100,
}

print("Sending request to OpenRouter...")
try:
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data,
        timeout=30,
    )
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"\nResponse Content:")
        print(content)
    else:
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
