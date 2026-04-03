from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_qa.config import load_config, project_root
from rag_qa.metrics import exact_match, token_f1
from rag_qa.pipeline import MultiAgentOrchestrator, build_index_from_corpus


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8", errors="ignore")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


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
    
    print("=== Agent Interpretability Trace ===")
    if res.trace:
        print(res.trace)
    else:
        print("No trace available.")
        
    print("\n=== Exact Final Answer ===")
    print(res.answer)
    
    if res.telemetry:
        print("\n=== Pipeline Telemetry ===")
        print(f"⏱️ Total Latency: {res.telemetry.get('total_latency', 0):.2f}s")
        print(f"🪙 Total Tokens: {res.telemetry.get('total_prompt_tokens', 0)} in, {res.telemetry.get('total_completion_tokens', 0)} out")
    
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

    # Prepare trace directory
    trace_dir = project_root() / "data" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    def run_arm(name: str, use_routing: bool) -> None:
        f1s = []
        ems = []
        
        # Telemetry & Audit aggregators
        arm_total_latency = 0.0
        arm_total_tokens = 0
        route_misclassifications = []
        
        arm_id = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").replace("/", "_").lower()
        
        # Load existing traces to support resume
        suffix = f"_{args.output_suffix}" if getattr(args, 'output_suffix', '') else ""
        trace_path = trace_dir / f"all_traces_{arm_id}{suffix}.json"
        
        if getattr(args, 'resume', False) and trace_path.exists():
            try:
                with open(trace_path, "r", encoding="utf-8") as tf:
                    all_traces = json.load(tf)
                print(f"[{name}] Resuming... Loaded {len(all_traces)} existing traces.")
            except Exception as e:
                print(f"[{name}] Failed to load traces for resume: {e}", file=sys.stderr)
                all_traces = []
        else:
            all_traces = []
        
        # Populate aggregators from loaded traces
        for t in all_traces:
            if t.get("gold") is not None:
                f1s.append(token_f1(t["exact_answer"], t["gold"]))
                ems.append(1.0 if exact_match(t["exact_answer"], t["gold"]) else 0.0)
            if use_routing and t.get("complexity") != "complex":
                route_misclassifications.append({"question": t["question"], "predicted": t["complexity"]})
            if t.get("telemetry"):
                arm_total_latency += t["telemetry"].get('total_latency', 0)
                arm_total_tokens += (t["telemetry"].get('total_prompt_tokens', 0) + t["telemetry"].get('total_completion_tokens', 0))

        completed_questions = {t["question"] for t in all_traces}
        
        for idx, r in enumerate(rows):
            q = r["question"]
            gold = r.get("gold_answer")
            
            if getattr(args, 'resume', False) and q in completed_questions:
                print(f"\n[{name}] Question {idx+1}/{len(rows)} (Skipped, already evaluated)")
                continue

            # Print explicit separation marker for real-time monitoring
            print(f"\n[{name}] Question {idx+1}/{len(rows)}: {q}")
            
            out = pipe.answer_question(q, use_routing=use_routing)
            
            # Aggregate Pipeline Telemetry
            if out.telemetry:
                arm_total_latency += out.telemetry.get('total_latency', 0)
                arm_total_tokens += (out.telemetry.get('total_prompt_tokens', 0) + out.telemetry.get('total_completion_tokens', 0))
            
            # Route Misclassification Audit
            # Assume eval_hotpot_complex contains only hard/complex questions
            if use_routing and out.complexity != "complex":
                alert = f"🚨 WARNING: Question misclassified as '{out.complexity.upper()}' instead of 'COMPLEX'!"
                print(f"\033[91m{alert}\033[0m")
                route_misclassifications.append({"question": q, "predicted": out.complexity})
            else:
                if use_routing:
                    print(f"✅ Route matched: COMPLEX")

            if gold is not None:
                f1s.append(token_f1(out.answer, gold))
                ems.append(1.0 if exact_match(out.answer, gold) else 0.0)
                
            rec = {
                "arm": name,
                "question": q,
                "complexity": out.complexity,
                "exact_answer": out.answer,
                "gold": gold,
                "trace": out.trace,
                "telemetry": out.telemetry
            }
            all_traces.append(rec)
            if args.dump:
                print(json.dumps(rec, ensure_ascii=False))
                
            # Log all traces to a single JSON file persistently on each loop
            # Removed the duplicate trace_path assignment here as it's defined above
            with open(trace_path, "w", encoding="utf-8") as tf:
                json.dump(all_traces, tf, ensure_ascii=False, indent=2)

        print(f"\n=== {name} ===")
        if f1s:
            print(f"Mean token-F1 (n={len(f1s)}): {sum(f1s)/len(f1s):.4f}")
            print(f"Exact match rate: {sum(ems)/len(ems):.4f}")
        else:
            print("No gold_answer fields — printed answers only." + (" Use --dump to see them." if not args.dump else ""))
            
        print(f"\n[Telemetry] Total Run Time: {arm_total_latency:.2f}s | Total Tokens Used: {arm_total_tokens}")
        
        if use_routing:
            print("\n[Route Misclassification Report]")
            if not route_misclassifications:
                print("Perfect Routing! 0 misclassifications.")
            else:
                print(f"Miss Rate: {len(route_misclassifications)}/{len(rows)} ({(len(route_misclassifications)/len(rows))*100:.1f}%)")
                for m in route_misclassifications:
                    print(f"- Question: '{m['question'][:60]}...' classified as {m['predicted'].upper()}")

    run_arm("Multi-Agent Orchestrator (RAG+Routing)", use_routing=True)
    # We optionally can disable baseline if we only want to run the advanced arm, but leaving it here for backward compatibility.
    # run_arm("Baseline (No retrieval/routing)", use_routing=False)
    
    print(f"\n✅ All interpretability traces saved to: {trace_dir}")
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
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    p = argparse.ArgumentParser(description="CDS547 Multi-Agent RAG Orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Ingest corpus and save index under data/index")

    pq = sub.add_parser("query", help="Ask one question via Orchestrator")
    pq.add_argument("question", type=str)
    pq.add_argument("--no-retrieval", action="store_true", help="Baseline: LLM without context or routing")

    pe = sub.add_parser("eval", help="Run JSONL eval set (Multi-Agent vs no retrieval)")
    pe.add_argument("questions", type=str, help="Path to .jsonl")
    pe.add_argument("--dump", action="store_true", help="Print one JSON per line per arm")
    pe.add_argument("--resume", action="store_true", help="Resume evaluation from existing traces file")
    pe.add_argument("--suffix", type=str, dest="output_suffix", default="", help="Suffix for output json files")

    pa = sub.add_parser("ablate-topk", help="Try several top_k on one question")
    pa.add_argument("question", type=str)
    pa.add_argument("--top-k", type=str, default="3,5,10", help="Comma-separated k values")

    args = p.parse_args()
    
    # Configure logging
    log_suffix = f"_{args.output_suffix}" if hasattr(args, 'output_suffix') and args.output_suffix else ""
    log_file_path = project_root() / "data" / f"console_output{log_suffix}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = TeeLogger(log_file_path)

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
