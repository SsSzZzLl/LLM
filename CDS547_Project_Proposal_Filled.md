# Group Project Proposal  
**CDS547 – Introduction to Large Language Models**  
**Date:** February 6, 2026  

---

## Project Title

**Retrieval-Augmented Generation (RAG) for Domain-Specific Document Question Answering**

---

## Group Members

| Student ID | Last Name | First Name | Role and Responsibilities |
|------------|-----------|------------|---------------------------|
| *[Replace]* | *[Replace]* | *[Replace]* | **Team Leader:** Coordinate tasks, milestones, and communication with instructor; ensure deliverables on time. |
| *[Replace]* | *[Replace]* | *[Replace]* | **Researcher:** Literature review on RAG, LLM evaluation, and retrieval methods; summarize findings for the group. |
| *[Replace]* | *[Replace]* | *[Replace]* | **Developer:** Implement document ingestion, embedding/retrieval pipeline, prompting, and LLM API or local inference integration. |
| *[Replace]* | *[Replace]* | *[Replace]* | **Evaluator:** Design evaluation protocol (metrics, test questions), run experiments, analyze results, draft report sections. |

*Note: Replace placeholder rows with actual members. Add or remove rows as needed.*

---

## Objectives

- To design and implement a **retrieval-augmented generation (RAG)** pipeline that answers user questions using a curated **domain-specific document corpus** (e.g., course materials, technical manuals, or a public benchmark dataset in one vertical domain).

- To compare **RAG-based answers** against **direct LLM generation without retrieval** on the same question set, measuring factuality, hallucination rate, and answer relevance.

- To evaluate the impact of **retrieval design choices** (e.g., chunking strategy, top-*k* retrieved passages, dense vs. sparse retrieval where feasible) on downstream answer quality.

- To document **limitations** (latency, cost, domain shift) and discuss **responsible use** of LLMs in knowledge-intensive settings.

---

## Expected Results

- A **working prototype** (code + short demo or notebook) that: ingests documents, builds a search index, retrieves relevant chunks, and generates answers with optional citation of source spans.

- **Quantitative comparison** on a fixed evaluation set: metrics such as exact match / token F1 (where reference answers exist), human or LLM-assisted scoring rubrics for faithfulness, and qualitative error analysis.

- A **written report** summarizing architecture, experiments, and findings, plus a **final presentation** aligned with course requirements.

- *(Optional stretch)* A small ablation comparing **two embedding or retrieval configurations** to support claims about design trade-offs.

---

## Tentative Timeline

**Project Timeline (Weeks 2–14)** — adjust dates to match your term calendar.

| Week range | Phase / Task |
|------------|----------------|
| 2–3 | **Phase 1: Proposal** — Finalize topic, corpus, and baseline plan. |
| 3–5 | **Literature Review** — RAG (Lewis et al.), LLM evaluation, retrieval and chunking best practices. |
| 5–10 | **Phase 2: Development** — Data prep, indexing, RAG pipeline, baseline “no retrieval” prompts. |
| 8–11 | **Experimentation** — Build eval set, run comparisons, log hyperparameters and failures. |
| 11–13 | **Phase 3: Finalization** — **Evaluation and analysis**, figures/tables, limitations. |
| 13–14 | **Final report and presentation** — Polish writing, rehearse demo. |

**Gantt-style summary (blocks span approximate weeks):**

```
        2  3  4  5  6  7  8  9 10 11 12 13 14
Proposal ███
Lit Rev     ██████
Development      ████████████
Experimentation           ████████
Eval & Analysis                    ██████
Final Report & Pres                    ████
```

---

## References

- Vaswani, A., et al. (2017). “Attention Is All You Need.” *NeurIPS*.

- Devlin, J., et al. (2019). “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” *ACL*.

- OpenAI. (2020). “Language Models are Few-Shot Learners.” *NeurIPS*.

- Lewis, P., et al. (2020). “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” *NeurIPS*.

- Gao, Y., et al. (2023). “Retrieval-Augmented Generation for Large Language Models: A Survey.” *(survey / arXiv — verify citation for your bibliography style).*

- *(Add)* Official docs for APIs or frameworks you use (e.g., LangChain, LlamaIndex, Hugging Face, OpenAI).

---

*End of proposal draft — export to PDF or paste into the official course template if required.*
