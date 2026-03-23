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

    # Support custom base URL for proxy/third-party APIs (e.g., OpenRouter)
    base_url = os.getenv("OPENAI_BASE_URL", None)
    
    # Create client with proper configuration
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    m = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # For OpenRouter, add required headers
    extra_headers = {}
    if base_url and "openrouter" in base_url.lower():
        # OpenRouter requires HTTP-Referer and X-Title headers
        extra_headers = {
            "HTTP-Referer": "https://github.com/llm-group/rag-qa",
            "X-Title": "CDS547 RAG QA Project",
        }
    
    resp = client.chat.completions.create(
        model=m,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_headers=extra_headers if extra_headers else None,
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
