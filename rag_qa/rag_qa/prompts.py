from __future__ import annotations

from typing import List, Tuple


def format_context_passages(passages: List[Tuple[str, str]]) -> str:
    """passages: list of (passage_id, text)"""
    lines = []
    for i, (pid, text) in enumerate(passages, start=1):
        lines.append(f"[{i}] (source: {pid})\n{text}")
    return "\n\n".join(lines)


def rag_system_prompt() -> str:
    return (
        "You are a careful assistant for document-grounded question answering. "
        "Answer using ONLY the provided passages. If the answer is not contained in them, say you cannot "
        "find it in the given context. When you use information from a passage, cite its bracket number, e.g. [1]."
    )


def rag_user_prompt(question: str, context_block: str) -> str:
    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely. Cite passage numbers in brackets where applicable."
    )


def no_context_user_prompt(question: str) -> str:
    return (
        "Answer the question to the best of your ability. If you are uncertain, say so.\n\n"
        f"Question: {question}"
    )


def no_context_system_prompt() -> str:
    return "You are a helpful assistant."
