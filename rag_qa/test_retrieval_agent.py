from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from rag_qa.agents.base_agent import AgentInput
from rag_qa.agents.retrieval_agent import RetrievalAgent
from rag_qa.index_store import DocumentIndex


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_index_dir(cfg: Dict[str, Any]) -> Path:
    index_dir = cfg.get("index_dir", "data/index")
    return (PROJECT_ROOT / index_dir).resolve()


def load_index(index_dir: Path) -> DocumentIndex:
    """
    这里是唯一最可能需要你按项目实际接口微调的地方。

    常见情况一般有三种：
    1. DocumentIndex.load(index_dir)
    2. DocumentIndex.from_dir(index_dir)
    3. DocumentIndex(...) 然后再 index.load(...)

    先按最常见的类方法写，如果报错，我下一步带你改成你的实际版本。
    """
    if hasattr(DocumentIndex, "load"):
        return DocumentIndex.load(index_dir)  # type: ignore[attr-defined]
    if hasattr(DocumentIndex, "from_dir"):
        return DocumentIndex.from_dir(index_dir)  # type: ignore[attr-defined]

    raise AttributeError(
        "DocumentIndex 没有 load / from_dir 方法。"
        "请把 rag_qa/index_store.py 里 DocumentIndex 的定义贴给我，我帮你对上。"
    )


def pretty_print_result(question: str, result) -> None:
    print("\n" + "=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Should retry: {result.should_retry}")

    metadata = result.metadata or {}
    print("\nMetadata:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))

    evidence: List[str] = result.evidence or []
    if not evidence:
        print("\nNo evidence retrieved.")
        return

    print(f"\nRetrieved {len(evidence)} passages:\n")
    for i, text in enumerate(evidence, start=1):
        print(f"[{i}] {text[:500]}")
        if len(text) > 500:
            print("...")

    passages = metadata.get("passages", [])
    scores = metadata.get("scores", [])

    if passages:
        print("\nPassage IDs:")
        for i, item in enumerate(passages, start=1):
            try:
                stable_id, _ = item
            except Exception:
                stable_id = str(item)
            score = scores[i - 1] if i - 1 < len(scores) else None
            print(f"  {i}. stable_id={stable_id}, score={score}")


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    index_dir = get_index_dir(cfg)

    print(f"Loading config from: {CONFIG_PATH}")
    print(f"Loading index from: {index_dir}")

    index = load_index(index_dir)
    agent = RetrievalAgent.from_config(index=index, cfg=cfg)

    test_questions = [
        "What does RAG condition the generator on?",
        "Who introduced RAG for knowledge-intensive tasks?",
        "What is hallucination in the context of LLM evaluation?",
        "What should teams document for reproducibility?",
        "Compare dense retrieval and BM25 for multi-hop QA.",
    ]

    for question in test_questions:
        # 普通单跳测试
        input_data = AgentInput(
            question=question,
            context=[],
            sub_questions=[],
            metadata={}
        )
        result = agent.run(input_data)
        pretty_print_result(question, result)

    # 额外加一个“多子问题”测试，顺手验证 _queries() + _merge_hits()
    multi_hop_question = "Compare dense retrieval and BM25 for multi-hop QA."
    multi_hop_input = AgentInput(
        question=multi_hop_question,
        context=[],
        sub_questions=[
            "What is dense retrieval?",
            "What is BM25?",
            "What are the trade-offs between dense retrieval and BM25?",
        ],
        metadata={
            "complexity": "complex",
            "use_multi_hop": True,
        },
    )
    multi_hop_result = agent.run(multi_hop_input)
    pretty_print_result("[MULTI-HOP] " + multi_hop_question, multi_hop_result)


if __name__ == "__main__":
    main()