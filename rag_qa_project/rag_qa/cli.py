from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_qa.agents.route_agent import (
    RouteAgent,
    RouteEvaluator,
    create_test_dataset,
)
from rag_qa.config import load_config, project_root
from rag_qa.metrics import exact_match, token_f1
from rag_qa.pipeline import RAGPipeline, build_index_from_corpus


def cmd_build(_: argparse.Namespace) -> int:
    out = build_index_from_corpus()
    print(f"Index saved to: {out}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipe = RAGPipeline.from_disk(use_routing=args.dynamic_routing)
    use_rag = not args.no_retrieval
    res = pipe.answer_question(args.question, use_retrieval=use_rag, use_dynamic_routing=args.dynamic_routing)
    
    # Display routing info if dynamic routing was used
    if args.dynamic_routing and res.route_decision:
        print("=== Routing Decision ===")
        print(f"Complexity: {res.route_decision.complexity.value}")
        print(f"Confidence: {res.route_decision.confidence:.2f}")
        print(f"Reasoning: {res.route_decision.reasoning}")
        print(f"Strategy: {res.route_decision.recommended_strategy}")
        print()
    
    print("=== Answer ===")
    print(res.answer)
    if res.passages:
        print("\n=== Retrieved (id, score) ===")
        for (pid, text), sc in zip(res.passages, res.scores):
            preview = text[:200].replace("\n", " ")
            print(f"- {pid}  score={sc:.4f}  |  {preview}...")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    path = Path(args.questions)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    pipe = RAGPipeline.from_disk()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    def run_arm(name: str, use_rag: bool) -> None:
        f1s = []
        ems = []
        for r in rows:
            q = r["question"]
            gold = r.get("gold_answer")
            out = pipe.answer_question(q, use_retrieval=use_rag)
            if gold is not None:
                f1s.append(token_f1(out.answer, gold))
                ems.append(1.0 if exact_match(out.answer, gold) else 0.0)
            rec = {
                "arm": name,
                "question": q,
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

    run_arm("RAG", use_rag=True)
    run_arm("No retrieval", use_rag=False)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Quick top-k ablation: rebuild not required; uses current index."""
    cfg = load_config()
    cfg_path = project_root() / "config.yaml"
    pipe = RAGPipeline.from_disk(cfg)

    ks = [int(x) for x in args.top_k.split(",")]
    q = args.question
    print(f"Question: {q}\n")
    for k in ks:
        cfg["retrieval"] = dict(cfg.get("retrieval", {}))
        cfg["retrieval"]["top_k"] = k
        pipe_k = RAGPipeline(pipe.index, cfg)
        res = pipe_k.answer_question(q, use_retrieval=True)
        print(f"--- top_k={k} ---")
        print(res.answer[:500] + ("..." if len(res.answer) > 500 else ""))
        print()
    return 0


def cmd_eval_route(_: argparse.Namespace) -> int:
    """Evaluate Route Agent classification accuracy."""
    print("=" * 60)
    print("Route Agent Evaluation")
    print("=" * 60)
    
    # Initialize RouteAgent
    cfg = load_config()
    g_cfg = cfg.get("generation", {})
    agent = RouteAgent(
        model=g_cfg.get("openai_model"),
        temperature=0.1,
        max_tokens=512,
    )
    
    # Load test dataset
    test_data = create_test_dataset()
    evaluator = RouteEvaluator()
    
    print(f"\nEvaluating on {len(test_data)} test questions...\n")
    
    for question, ground_truth in test_data:
        decision = agent.classify(question)
        evaluator.add_prediction(question, decision, ground_truth)
        
        status = "✓" if decision.complexity == ground_truth else "✗"
        print(f"{status} [{decision.complexity.value:8}] {question[:60]}...")
    
    # Calculate and display metrics
    metrics = evaluator.evaluate()
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Average Confidence: {metrics['avg_confidence']:.2%}")
    print(f"Total Samples: {metrics['total_samples']}")
    
    print("\nPer-Class Metrics:")
    for cls, cls_metrics in metrics['per_class_metrics'].items():
        print(f"\n  {cls.upper()}:")
        print(f"    Precision: {cls_metrics['precision']:.2%}")
        print(f"    Recall: {cls_metrics['recall']:.2%}")
        print(f"    F1-Score: {cls_metrics['f1']:.2%}")
        print(f"    Support: {cls_metrics['support']}")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"{'Predicted →':15} simple    moderate  complex")
    for true_cls in ['simple', 'moderate', 'complex']:
        row = f"{true_cls:15}"
        for pred_cls in ['simple', 'moderate', 'complex']:
            row += f"{cm[pred_cls][true_cls]:10}"
        print(row)
    
    # Show misclassified examples
    misclassified = evaluator.get_misclassified()
    if misclassified:
        print("\n" + "=" * 60)
        print("Misclassified Examples")
        print("=" * 60)
        for item in misclassified[:5]:  # Show first 5
            print(f"\nQuestion: {item['question']}")
            print(f"  Predicted: {item['predicted']} (conf: {item['confidence']:.2f})")
            print(f"  Ground Truth: {item['ground_truth']}")
            print(f"  Reasoning: {item['reasoning'][:100]}...")
    
    return 0


def cmd_route_demo(_: argparse.Namespace) -> int:
    """Interactive demo of Route Agent classification."""
    print("=" * 60)
    print("Route Agent Interactive Demo")
    print("=" * 60)
    print("\nEnter questions to see how they are routed.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    cfg = load_config()
    g_cfg = cfg.get("generation", {})
    agent = RouteAgent(
        model=g_cfg.get("openai_model"),
        temperature=0.1,
        max_tokens=512,
    )
    
    while True:
        try:
            question = input("> ").strip()
            if question.lower() in ('quit', 'exit', 'q'):
                break
            if not question:
                continue
            
            decision, config = agent.route(question)
            
            print(f"\n  Complexity: {decision.complexity.value.upper()}")
            print(f"  Confidence: {decision.confidence:.2%}")
            print(f"  Strategy: {config['strategy_name']}")
            print(f"  Use Retrieval: {config['use_retrieval']}")
            print(f"  Use Multi-Hop: {config['use_multi_hop']}")
            print(f"  Reasoning: {decision.reasoning}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            break
    
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CDS547 RAG QA pipeline with Dynamic Routing")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Ingest corpus and save index under data/index")

    pq = sub.add_parser("query", help="Ask one question")
    pq.add_argument("question", type=str)
    pq.add_argument("--no-retrieval", action="store_true", help="Baseline: LLM without context")
    pq.add_argument("--dynamic-routing", action="store_true", help="Enable dynamic routing via RouteAgent")

    pe = sub.add_parser("eval", help="Run JSONL eval set (RAG vs no retrieval)")
    pe.add_argument("questions", type=str, help="Path to .jsonl")
    pe.add_argument("--dump", action="store_true", help="Print one JSON per line per arm")

    pa = sub.add_parser("ablate-topk", help="Try several top_k on one question (edit config temporarily in-memory)")
    pa.add_argument("question", type=str)
    pa.add_argument("--top-k", type=str, default="3,5,10", help="Comma-separated k values")
    
    # Route Agent commands
    sub.add_parser("eval-route", help="Evaluate Route Agent classification accuracy")
    sub.add_parser("route-demo", help="Interactive demo of Route Agent question classification")

    args = p.parse_args()
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "query":
        return cmd_query(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "ablate-topk":
        return cmd_ablate(args)
    if args.cmd == "eval-route":
        return cmd_eval_route(args)
    if args.cmd == "route-demo":
        return cmd_route_demo(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
