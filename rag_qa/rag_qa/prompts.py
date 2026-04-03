from __future__ import annotations

from typing import List, Tuple


def format_context_passages(passages: List[Tuple[str, str]]) -> str:
    """passages: list of (passage_id, text)"""
    lines = []
    for i, (pid, text) in enumerate(passages, start=1):
        lines.append(f"[{i}] (source: {pid})\n{text}")
    return "\n\n".join(lines)


def json_format_instructions() -> str:
    return (
        "IMPORTANT: You MUST return your answer strictly as a raw JSON object. "
        "Do NOT wrap it in markdown code blocks (e.g., no ```json). "
        "Your JSON MUST conform exactly to the following structure:\n"
        "{\n"
        '  "exact_answer": "<Extremely concise final answer. ONLY the exact entity name, number, or yes/no. NO conversational text at all>",\n'
        '  "thought_process": "<Detailed step-by-step reasoning explaining how you arrived at the answer. EXTREMELY CRITICAL: MUST KEEP UNDER 3 SENTENCES OR 50 WORDS. DO NOT COPY PASTE PASSAGES.>",\n'
        '  "citations": ["<array of passage bracket IDs used, if any>"]\n'
        "}"
    )


def rag_system_prompt() -> str:
    return (
        "You are a careful assistant for document-grounded question answering. "
        "Answer using ONLY the provided passages. If the answer is not contained in them, say you cannot "
        "find it in the given context. " + json_format_instructions()
    )


def rag_user_prompt(question: str, context_block: str) -> str:
    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Analyze the provided context and answer the question using the requested JSON format."
    )


def no_context_user_prompt(question: str) -> str:
    return (
        "Answer the question to the best of your ability. If you are uncertain, say so.\n\n"
        f"Question: {question}\n\n"
        "Analyze the question and provide your response using the requested JSON format."
    )


def no_context_system_prompt() -> str:
    return (
        "You are a helpful answering assistant. " + json_format_instructions()
    )

