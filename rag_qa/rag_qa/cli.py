from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_qa.config import load_config, project_root
from rag_qa.metrics import exact_match, token_f1
from rag_qa.pipeline import MultiAgentOrchestrator, build_index_from_corpus


def cmd_build(_: argparse.Namespace) -> int:
    out = build_index_from_corpus()
    print(f"Index saved to: {out}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipe = MultiAgentOrchestrator.from_disk()
    
    # --no-retrieval disables the routing capabilities and acts as a strict LLM-only baseline
    use_routing = not args.no_retrieval
    
    print("\n[Orchestrator Processing...]")
    res = pipe.answer_question(args.question, use_routing=use_routing)
    
    print("=== Routing Decision ===")
    print(f"Complexity: {res.complexity.upper()}")
    print(f"Strategy:   {res.routing_strategy}\n")
    
    print("=== Final Answer ===")
    print(res.answer)
    
    if res.passages:
        print("\n=== Retrieved Evidence (id, score) ===")
        for (pid, text), sc in zip(res.passages, res.scores):
            preview = text[:200].replace("\n", " ")
            print(f"- [{pid}] score={sc:.4f} | {preview}...")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    path = Path(args.questions)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    pipe = MultiAgentOrchestrator.from_disk()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    def run_arm(name: str, use_routing: bool) -> None:
        f1s = []
        ems = []
        for r in rows:
            q = r["question"]
            gold = r.get("gold_answer")
            out = pipe.answer_question(q, use_routing=use_routing)
            if gold is not None:
                f1s.append(token_f1(out.answer, gold))
                ems.append(1.0 if exact_match(out.answer, gold) else 0.0)
            rec = {
                "arm": name,
                "question": q,
                "complexity": out.complexity,
                "answer": out.answer,
                "gold": gold,
            }
            if args.dump:
                print(json.dumps(rec, ensure_ascii=False))

        print(f"\n=== {name} ===")
        if f1s:
            print(f"Mean token-F1 (n={len(f1s)}): {sum(f1s)/len(f1s):.4f}")
            print(f"Exact match rate: {sum(ems)/len(ems):.4f}")
        else:
            print("No gold_answer fields — printed answers only." + (" Use --dump to see them." if not args.dump else ""))

    run_arm("Multi-Agent Orchestrator (RAG+Routing)", use_routing=True)
    run_arm("Baseline (No retrieval/routing)", use_routing=False)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Quick top-k ablation: rebuild not required; uses current index."""
    cfg = load_config()
    cfg_path = project_root() / "config.yaml"
    pipe = MultiAgentOrchestrator.from_disk(cfg)

    ks = [int(x) for x in args.top_k.split(",")]
    q = args.question
    print(f"Question: {q}\n")
    for k in ks:
        cfg["retrieval"] = dict(cfg.get("retrieval", {}))
        cfg["retrieval"]["top_k"] = k
        pipe_k = MultiAgentOrchestrator(pipe.index, cfg)
        res = pipe_k.answer_question(q, use_routing=True)
        print(f"--- top_k={k} | Complexity={res.complexity} ---")
        print(res.answer[:500] + ("..." if len(res.answer) > 500 else ""))
        print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CDS547 Multi-Agent RAG Orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Ingest corpus and save index under data/index")

    pq = sub.add_parser("query", help="Ask one question via Orchestrator")
    pq.add_argument("question", type=str)
    pq.add_argument("--no-retrieval", action="store_true", help="Baseline: LLM without context or routing")

    pe = sub.add_parser("eval", help="Run JSONL eval set (Multi-Agent vs no retrieval)")
    pe.add_argument("questions", type=str, help="Path to .jsonl")
    pe.add_argument("--dump", action="store_true", help="Print one JSON per line per arm")

    pa = sub.add_parser("ablate-topk", help="Try several top_k on one question")
    pa.add_argument("question", type=str)
    pa.add_argument("--top-k", type=str, default="3,5,10", help="Comma-separated k values")

    args = p.parse_args()
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "query":
        return cmd_query(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "ablate-topk":
        return cmd_ablate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
