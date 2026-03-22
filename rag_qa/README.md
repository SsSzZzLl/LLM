# RAG Document QA — Technical Stack (CDS547)

End-to-end pipeline: **ingest `.txt` / `.md` → chunk → BM25 + dense embeddings → retrieve → prompt → OpenAI chat (or mock)**. Includes **eval** script for RAG vs. no-retrieval and **top-k ablation**.

## Setup (Windows / macOS / Linux)

```bash
cd rag_qa
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and add `OPENAI_API_KEY` for real LLM answers. Without it, generation returns a **mock** answer (retrieval still runs).

First run downloads `sentence-transformers/all-MiniLM-L6-v2` (~80MB).

## Configuration

Edit `config.yaml`:

| Key | Role |
|-----|------|
| `corpus_dir` | Folder of `.txt` / `.md` documents |
| `index_dir` | Where the built index is stored |
| `chunking.max_chars` / `overlap_chars` | Window size and overlap |
| `retrieval.mode` | `dense`, `bm25`, or `hybrid` |
| `retrieval.top_k` | Passages fed to the LLM |
| `embedding.model_name` | Sentence-Transformers model |
| `generation.*` | OpenAI model name, temperature, max_tokens |

## Commands

```bash
# Build index from corpus (after editing docs under data/corpus)
python -m rag_qa build

# Single question
python -m rag_qa query "What is RAG?"
python -m rag_qa query "What is attention?" --no-retrieval

# Eval set: JSONL lines with "question" and optional "gold_answer"
python -m rag_qa eval eval/questions.jsonl

# Ablation: same question, different top_k (in-memory only)
python -m rag_qa ablate-topk "Who introduced RAG?" --top-k 2,5,10
```

CLI entry point (after `pip install -e .`): `rag-qa build` etc.

## Project layout

```
rag_qa/
  config.yaml
  rag_qa/
    chunking.py      # character windows
    ingest.py        # load corpus
    index_store.py   # BM25 + dense matrix, save/load
    retrieve.py      # dense | bm25 | hybrid
    prompts.py       # RAG vs no-context prompts
    generate.py      # OpenAI + mock fallback
    pipeline.py      # orchestration
    metrics.py       # token F1, exact match
    cli.py
  data/corpus/       # put your domain documents here
  eval/questions.jsonl
```

## For your report

- **Baselines:** `query --no-retrieval` vs default RAG.
- **Ablations:** `retrieval.mode` (`bm25` / `dense` / `hybrid`), `chunking`, `top_k` via `ablate-topk` or config + rebuild.
- **Metrics:** extend `metrics.py` or log LLM-judge scores; `eval` prints mean token-F1 and EM when `gold_answer` is present.
- **Reproducibility:** save `config.yaml`, model IDs, and index build date in an appendix.

## Optional: larger corpus

Replace `data/corpus` with PDFs converted to text (not included). Add cleaning steps in `ingest.py` if needed.
