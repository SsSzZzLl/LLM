from __future__ import annotations

import os
import time
from typing import List

from dotenv import load_dotenv

load_dotenv()


def generate_chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    tracker: dict | None = None,
) -> str:
    """OpenAI Chat Completions. Falls back to mock if no API key."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _mock_answer(user)

    from openai import OpenAI

    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    m = model or os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo")
    
    start_time = time.time()
    resp = client.chat.completions.create(
        model=m,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    latency = time.time() - start_time
    
    if tracker is not None:
        tracker["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
        tracker["completion_tokens"] = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        tracker["total_tokens"] = getattr(resp.usage, "total_tokens", 0) if resp.usage else 0
        tracker["latency"] = latency
        tracker["model"] = m
        
        # Real-time console logging if specified in the tracker
        agent_name = tracker.get("agent_name", "LLM Call")
        print(f"[{agent_name}] ⏱️ {latency:.2f}s | 🪙 {tracker['prompt_tokens']} in, {tracker['completion_tokens']} out")

    return (resp.choices[0].message.content or "").strip()


def _mock_answer(user: str) -> str:
    return (
        "[MOCK — set OPENAI_API_KEY for real LLM output] "
        "The pipeline ran without an API key. Retrieved context was passed in the user message; "
        "for the course demo, configure .env and re-run. "
        "Preview of prompt tail:\n---\n"
        + user[-800:]
    )
