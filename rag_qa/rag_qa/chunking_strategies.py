"""
多种 Chunking 策略实现
包含：字符级、Token级、语义分块、LangChain递归分块
"""
from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Optional


@dataclass
class Chunk:
    text: str
    source_id: str
    chunk_index: int
    strategy: str = ""  # 记录使用的分块策略


# ==================== 1. 字符级分块（基础滑动窗口） ====================

def chunk_text_sliding_window(
    text: str,
    source_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """字符级滑动窗口分块"""
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    step = max(1, max_chars - overlap_chars)

    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(
                text=piece,
                source_id=source_id,
                chunk_index=idx,
                strategy="sliding_window"
            ))
            idx += 1
        if end >= len(text):
            break
        start += step

    return chunks


# ==================== 2. Token级分块 ====================

def chunk_text_token_level(
    text: str,
    source_id: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",  # OpenAI的编码
) -> List[Chunk]:
    """Token级分块，使用tiktoken"""
    try:
        import tiktoken
    except ImportError:
        raise ImportError("请安装 tiktoken: pip install tiktoken")
    
    # 获取编码器
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except:
        encoding = tiktoken.get_encoding("gpt2")  # 回退到gpt2编码
    
    # 编码文本为token
    tokens = encoding.encode(text)
    
    if not tokens:
        return []
    
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    step = max(1, max_tokens - overlap_tokens)
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        token_chunk = tokens[start:end]
        
        # 解码回文本
        try:
            piece = encoding.decode(token_chunk).strip()
        except:
            piece = text[start:end]  # 回退到字符级
        
        if piece:
            chunks.append(Chunk(
                text=piece,
                source_id=source_id,
                chunk_index=idx,
                strategy="token_level"
            ))
            idx += 1
        
        if end >= len(tokens):
            break
        start += step
    
    return chunks


# ==================== 3. 语义分块（基于句子嵌入相似度） ====================

def chunk_text_semantic(
    text: str,
    source_id: str,
    max_sentences: int = 10,
    similarity_threshold: float = 0.7,
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> List[Chunk]:
    """
    语义分块：基于句子嵌入相似度进行分组
    将语义相似的句子组合在一起
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError("请安装依赖: pip install sentence-transformers scikit-learn")
    
    # 分割句子
    sentences = split_sentences(text)
    if not sentences:
        return []
    
    if len(sentences) <= max_sentences:
        return [Chunk(
            text=" ".join(sentences),
            source_id=source_id,
            chunk_index=0,
            strategy="semantic"
        )]
    
    # 加载模型
    try:
        model = SentenceTransformer(model_name)
    except:
        # 如果模型下载失败，回退到字符级
        return chunk_text_sliding_window(text, source_id)
    
    # 计算句子嵌入
    embeddings = model.encode(sentences)
    
    # 基于相似度进行分组
    chunks: List[Chunk] = []
    current_group = [sentences[0]]
    current_embeddings = [embeddings[0]]
    idx = 0
    
    for i in range(1, len(sentences)):
        # 计算当前句子与组内句子的平均相似度
        similarities = cosine_similarity(
            [embeddings[i]],
            current_embeddings
        )[0]
        avg_similarity = np.mean(similarities)
        
        # 如果相似度足够高且未达到最大句子数，加入当前组
        if avg_similarity >= similarity_threshold and len(current_group) < max_sentences:
            current_group.append(sentences[i])
            current_embeddings.append(embeddings[i])
        else:
            # 保存当前组，开始新组
            chunks.append(Chunk(
                text=" ".join(current_group),
                source_id=source_id,
                chunk_index=idx,
                strategy="semantic"
            ))
            idx += 1
            current_group = [sentences[i]]
            current_embeddings = [embeddings[i]]
    
    # 保存最后一组
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            source_id=source_id,
            chunk_index=idx,
            strategy="semantic"
        ))
    
    return chunks


def split_sentences(text: str) -> List[str]:
    """分割文本为句子"""
    # 支持中英文句子分割
    sentence_pattern = r'[^。！？.!?]+[。！？.!?]+'
    sentences = re.findall(sentence_pattern, text)
    
    # 如果没有找到句子（比如没有标点），按行分割
    if not sentences:
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    return sentences


# ==================== 4. LangChain RecursiveCharacterTextSplitter ====================

def chunk_text_langchain_recursive(
    text: str,
    source_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """使用 LangChain 的 RecursiveCharacterTextSplitter"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # 回退到字符级滑动窗口
        return chunk_text_sliding_window(text, source_id, chunk_size, chunk_overlap)
    
    # 针对保险/法律文档优化分隔符
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
        keep_separator=True,
    )
    
    documents = text_splitter.create_documents([text])
    
    chunks = []
    for idx, doc in enumerate(documents):
        chunks.append(Chunk(
            text=doc.page_content,
            source_id=source_id,
            chunk_index=idx,
            strategy="langchain_recursive"
        ))
    
    return chunks


# ==================== 5. 固定大小分块（按段落优先） ====================

def chunk_text_paragraph_based(
    text: str,
    source_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """基于段落的分块，尽量保持段落完整"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return []
    
    chunks: List[Chunk] = []
    current_chunk = []
    current_length = 0
    idx = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # 如果当前段落加上已有内容超过限制，保存当前chunk
        if current_length + para_length > max_chars and current_chunk:
            chunks.append(Chunk(
                text="\n\n".join(current_chunk),
                source_id=source_id,
                chunk_index=idx,
                strategy="paragraph_based"
            ))
            idx += 1
            
            # 保留重叠部分
            overlap_text = "\n\n".join(current_chunk)[-overlap_chars:]
            current_chunk = [overlap_text, para] if overlap_chars > 0 else [para]
            current_length = len(overlap_text) + para_length
        else:
            current_chunk.append(para)
            current_length += para_length
    
    # 保存最后一个chunk
    if current_chunk:
        chunks.append(Chunk(
            text="\n\n".join(current_chunk),
            source_id=source_id,
            chunk_index=idx,
            strategy="paragraph_based"
        ))
    
    return chunks


# ==================== 策略注册表 ====================

CHUNKING_STRATEGIES = {
    "sliding_window": chunk_text_sliding_window,
    "token_level": chunk_text_token_level,
    "semantic": chunk_text_semantic,
    "langchain_recursive": chunk_text_langchain_recursive,
    "paragraph_based": chunk_text_paragraph_based,
}


def chunk_document(
    text: str,
    source_id: str,
    strategy: str = "langchain_recursive",
    **kwargs
) -> List[Chunk]:
    """
    统一的文档分块接口
    
    Args:
        text: 文档文本
        source_id: 文档标识
        strategy: 分块策略名称
        **kwargs: 策略特定参数
    
    Returns:
        List[Chunk]: 分块结果
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"未知的分块策略: {strategy}。可用策略: {list(CHUNKING_STRATEGIES.keys())}")
    
    chunking_func = CHUNKING_STRATEGIES[strategy]
    return chunking_func(text, source_id, **kwargs)
