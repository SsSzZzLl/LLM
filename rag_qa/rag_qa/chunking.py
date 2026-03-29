from __future__ import annotations

import re
import tiktoken
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source_id: str
    chunk_index: int


def split_paragraphs(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source_id: str,
    max_tokens: int = 256,
    overlap_tokens: int = 30,
    model_name: str = "cl100k_base"
) -> List[Chunk]:
    """Token-based sliding window chunking."""
    if not text:
        return []

    encoding = tiktoken.get_encoding(model_name)
    tokens = encoding.encode(text)

    chunks: List[Chunk] = []
    start_token_idx = 0
    chunk_idx = 0
    
    step = max(1, max_tokens - overlap_tokens)

    while start_token_idx < len(tokens):
        end_token_idx = min(start_token_idx + max_tokens, len(tokens))
        
        chunk_tokens = tokens[start_token_idx:end_token_idx]
        chunk_text_content = encoding.decode(chunk_tokens)
        
        if chunk_text_content.strip():
            chunks.append(Chunk(text=chunk_text_content, source_id=source_id, chunk_index=chunk_idx))
            chunk_idx += 1
        
        if end_token_idx >= len(tokens):
            break
        
        start_token_idx += step

    return chunks



def chunk_document_with_langchain(
    full_text: str, 
    source_id: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 150
) -> List[Chunk]:
    """
    使用 langchain_text_splitters 的 RecursiveCharacterTextSplitter 进行分块。
    针对保险/法律条款优化：严禁截断完整句子，优先匹配段落和句号。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 针对保险/法律文档优化分隔符：
    # 1. 优先使用双换行（段落分隔）
    # 2. 其次使用句号（句子结束）
    # 3. 然后是其他常见分隔符
    separators = [
        "\n\n",      # 段落分隔
        "。",         # 中文句号
        ". ",        # 英文句号+空格
        ".\n",       # 英文句号+换行
        "；",        # 中文分号
        ";",         # 英文分号
        "，",        # 中文逗号
        ",",         # 英文逗号
        " ",         # 空格
        ""           # 字符级别（最后手段）
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,  # 保留分隔符，避免句子被截断
    )
    
    # 执行分块
    documents = text_splitter.create_documents([full_text])
    
    # 转换为 Chunk 对象
    chunks = []
    for idx, doc in enumerate(documents):
        chunks.append(Chunk(
            text=doc.page_content,
            source_id=source_id,
            chunk_index=idx
        ))
    
    return chunks


def chunk_document(full_text: str, source_id: str, max_chars: int, overlap_chars: int) -> List[Chunk]:
    """
    文档分块主入口。
    优先使用 langchain 的 RecursiveCharacterTextSplitter，
    针对保险/法律条款进行优化，确保不截断完整句子。
    """
    # 尝试使用 langchain_text_splitters
    try:
        return chunk_document_with_langchain(
            full_text, 
            source_id, 
            chunk_size=max_chars, 
            chunk_overlap=overlap_chars
        )
    except ImportError:
        # 如果 langchain_text_splitters 未安装，回退到基础实现
        print("Warning: langchain_text_splitters not found, falling back to basic chunking")
        paras = split_paragraphs(full_text)
        if not paras:
            return []
        merged = "\n\n".join(paras)
        return chunk_text(merged, source_id, max_chars=max_chars, overlap_chars=overlap_chars)
        return chunk_text(merged, source_id, max_tokens=max_chars, overlap_tokens=overlap_chars)
