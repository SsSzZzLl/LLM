"""检查环境变量"""
import os
from dotenv import load_dotenv

# 先加载 .env
load_dotenv()

print("Environment Variables:")
print(f"OPENAI_API_KEY: {'*' * 20}{os.getenv('OPENAI_API_KEY', 'NOT SET')[-10:] if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT SET')}")
print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'NOT SET')}")
