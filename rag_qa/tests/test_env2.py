"""检查环境变量 - 详细版本"""
import os

# 不使用 load_dotenv，直接读取文件
env_path = ".env"
print(f"Reading {env_path} directly:")
print("-" * 50)

try:
    with open(env_path, 'r') as f:
        content = f.read()
        print(content)
except Exception as e:
    print(f"Error reading file: {e}")

print("-" * 50)
print("\nNow loading with dotenv:")
from dotenv import load_dotenv
load_dotenv(override=True)  # 强制覆盖

print(f"OPENAI_API_KEY loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"Value: {os.getenv('OPENAI_API_KEY', 'NOT SET')}")
