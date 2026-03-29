from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def generate_chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """OpenAI Chat Completions. Falls back to mock if no API key."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _mock_answer(user)

    from openai import OpenAI

    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    m = model or os.getenv("OPENAI_MODEL", "openai/gpt-3.5-turbo")
    resp = client.chat.completions.create(
        model=m,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _mock_answer(user: str) -> str:
    return (
        "[MOCK — set OPENAI_API_KEY for real LLM output] "
        "The pipeline ran without an API key. Retrieved context was passed in the user message; "
        "for the course demo, configure .env and re-run. "
        "Preview of prompt tail:\n---\n"
        + user[-800:]
    )
