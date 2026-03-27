"""调试 API 响应"""
import os
from dotenv import load_dotenv

load_dotenv()

from rag_qa.generate import generate_chat

# 测试简单的 API 调用
system = "You are a helpful assistant. Reply with a short answer."
user = "What is 2+2? Answer in JSON format: {\"answer\": number}"

print("Testing API connection...")
print(f"Base URL: {os.getenv('OPENAI_BASE_URL', 'default')}")
print(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
print()

try:
    response = generate_chat(system, user, temperature=0.1, max_tokens=100)
    print("API Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
