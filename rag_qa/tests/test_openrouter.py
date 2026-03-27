"""直接测试 OpenRouter API"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "").strip()
base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

print(f"API Key: {api_key[:20]}...{api_key[-10:]}")
print(f"Base URL: {base_url}")
print(f"Model: {model}")
print()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/llm-group/rag-qa",
    "X-Title": "CDS547 RAG QA Project",
}

data = {
    "model": model,
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
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
