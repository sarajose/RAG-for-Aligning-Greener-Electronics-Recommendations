# Commands Reference — RAG Pipeline for Greener Electronics

> **Prerequisites**: activate the virtualenv first.
> ```powershell
> .\venv\Scripts\Activate.ps1
> ```
>
> All commands are run from the repository root.  
> Chunking (`retrieval/chunking_evidence.py`) is **already done** — the files
> `outputs/evidence.csv` and `outputs/evidence_recommendation.csv` already exist.
> Start from **Step 2** if indices are already built, or **Step 3** if you only
> want to re-run retrieval and later steps.

---

## Overview

| Step | What it does | Estimated time |
|------|-------------|----------------|
| 1 | *(Done)* Chunk HTML evidence → CSV | ~5 min |
| 2 | Build FAISS + BM25 indices | ~10–20 min (first run) |
| 3 | Evaluate retrieval — gold standard | ~2–5 min |
| 4 | Evaluate retrieval — MTEB LegalBench | ~5–15 min (first run includes download) |
| 5 | Classify whitepaper recommendations | ~2–4 h (GPU) |
| 6 | LLM-as-judge on classifications | ~1–2 h (GPU) |
| 7 | Full pipeline (steps 3+5+6 combined) | ~3–6 h (GPU) |

---

## Step 1 — Chunking (already done, skip if CSVs exist)

```powershell
# Evidence HTML → outputs/evidence.csv
python retrieval/chunking_evidence.py `
    -i data/evidence `
    -o outputs/evidence.csv

# Recommendation stub CSV is already at:
#   data/recommendations_whitepaper/recommendations_empty.csv
```

---

## Step 2 — Build Search Indices

Build the FAISS (dense) and BM25 (sparse) indices from the evidence CSV.
The indices are saved to `outputs/indices/` and only need to be built once.

```powershell
python main.py build `
    -i outputs/evidence.csv `
    -m bge-m3
```

**Outputs**:
- `outputs/indices/bge-m3_faiss.index`
- `outputs/indices/bge-m3_bm25.pkl`
- `outputs/indices/bge-m3_chunks.pkl`

> Use `-m minilm` for a faster/lighter run during development.

---

## Step 3 — Evaluate Retrieval (Gold Standard)

Evaluates document-level and paragraph-level retrieval quality against the
hand-annotated gold standard CSV using Hit@k, Recall@k, MRR, MAP, NDCG@k at
k ∈ {1, 3, 5, 10, 20}.

```powershell
python main.py evaluate `
    --gold data/gold_standard_doc_level/gold_standard.csv `
    -o outputs/metrics_retrieval.json
```

Optional flags:
```powershell
python main.py evaluate `
    --gold data/gold_standard_doc_level/gold_standard.csv `
    --no-rerank `          # skip cross-encoder reranker (faster)
    -k 20 `               # top-k for retrieval
    --rerank-top 10 `     # how many to keep after reranking
    -o outputs/metrics_retrieval.json
```

**Outputs**:
- Console: formatted retrieval report
- `outputs/metrics_retrieval.json`: metric values as JSON

---

## Step 4 — Evaluate Retrieval on MTEB LegalBench

This project uses **`mteb/legalbench_consumer_contracts_qa`** as the external
benchmark. The dataset is downloaded automatically from Hugging Face on first run.

```powershell
python scripts/run_mteb_legalbench_eval.py `
    --dataset mteb/legalbench_consumer_contracts_qa `
    --split test `
    --model bge-m3
```

**Outputs**:
- `outputs/mteb_legalbench_metrics.csv`
- `outputs/mteb_legalbench_metrics.json`

> For a quick test run, add `--max-corpus 10000`.

---

## Step 5 — Classify Whitepaper Recommendations

Retrieves top-k EU law chunks for each whitepaper recommendation and
classifies the alignment label using `Qwen/Qwen2.5-7B-Instruct` (local, ~15 GB).

```powershell
python main.py whitepaper `
    -i data/recommendations_whitepaper/recommendations_empty.csv `
    -o outputs/whitepaper_classified.csv
```

**Outputs**:
- `outputs/whitepaper_classified.csv` — one row per recommendation with
  `alignment_label`, `justification`, `cited_chunk_ids`, `top_chunk_texts`,
  `retrieved_documents`, `human_label` (empty), `human_notes` (empty)
- `outputs/whitepaper_classified_retrieved_chunks.csv` — one row per
  (recommendation, chunk) pair with full chunk text for human evaluation

> Add `--retrieve-only` to skip the LLM and only export retrieval results.

---

## Step 6 — LLM-as-Judge on Classifications

Re-run whitepaper classification **and** evaluate each result with the
LLM-as-judge (`mistralai/Mistral-7B-Instruct-v0.3`).  The judge scores
label correctness, justification quality, and evidence usage (1–5 each).

```powershell
python main.py whitepaper `
    -i data/recommendations_whitepaper/recommendations_empty.csv `
    -o outputs/whitepaper_classified.csv `
    --judge
```

**Additional output**:
- `outputs/whitepaper_classified_judge.csv` — judge scores and reasoning per recommendation

---

## Step 7 — Full Pipeline (Retrieval → Classification → Judge)

Runs retrieval evaluation on the gold standard, then classifies the
whitepaper recommendations, and finally runs the LLM-as-judge — all in
one command.

```powershell
python main.py run `
    --gold data/gold_standard_doc_level/gold_standard.csv `
    -i data/recommendations_whitepaper/recommendations_empty.csv `
    -o outputs/classified.csv `
    --judge
```

**Outputs** (in `outputs/`):
| File | Contents |
|------|---------|
| `classified.csv` | Alignment labels + justifications |
| `classified_retrieved_chunks.csv` | k retrieved chunks per recommendation |
| `judge_results.csv` | LLM-as-judge scores per classification |
| `metrics.json` | Retrieval + classification metrics |

---

## Script-First Evaluation (Recommended for Lower Memory)

Run evaluations as Python scripts and save compact CSV/JSON outputs.
Then use a notebook only for plotting.

### A) Small Gold Standard (Document-Level Only)

```powershell
python scripts/run_gold_doc_eval.py `
    --gold data/gold_standard_doc_level/gold_standard.csv `
    --model bge-m3 `
    --top-k 20 `
    --rerank-top 10
```

Outputs:
- `outputs/gold_doc_eval_metrics.csv`
- `outputs/gold_doc_eval_metrics.json`

### B) External MTEB Benchmark (Legal)

This project uses **`mteb/legalbench_consumer_contracts_qa`**.
The dataset is downloaded automatically from Hugging Face at first run.

```powershell
python scripts/run_mteb_legalbench_eval.py `
    --dataset mteb/legalbench_consumer_contracts_qa `
    --split test `
    --model bge-m3
```

Outputs:
- `outputs/mteb_legalbench_metrics.csv`
- `outputs/mteb_legalbench_metrics.json`

> For a quick test run, add `--max-corpus 10000`.

## Notebook Workflow (Visualization Only)

Use this notebook for plots only (no retrieval/indexing inside notebook):

| Notebook | Purpose |
|----------|---------|
| `notebooks/05_eval_visualizations_only.ipynb` | Visualize saved gold + MTEB evaluation metrics |

---

## Quick-Reference: All Output Files

| File | Generated by | Description |
|------|-------------|-------------|
| `outputs/evidence.csv` | chunking_evidence.py | All EU legislation chunks |
| `outputs/indices/bge-m3_*.index/.pkl` | `main.py build` | FAISS + BM25 indices |
| `outputs/metrics_retrieval.json` | `main.py evaluate` | Retrieval metrics (gold std) |
| `outputs/whitepaper_classified.csv` | `main.py whitepaper` | Alignment results for human eval |
| `outputs/whitepaper_classified_retrieved_chunks.csv` | `main.py whitepaper` | k retrieved chunks per rec |
| `outputs/whitepaper_classified_judge.csv` | `main.py whitepaper --judge` | LLM judge scores |
| `outputs/classified.csv` | `main.py run` | Full pipeline alignment results |
| `outputs/classified_retrieved_chunks.csv` | `main.py run` | k retrieved chunks per rec |
| `outputs/judge_results.csv` | `main.py run --judge` | LLM judge scores |
| `outputs/retrieval_comparison.csv` | Notebook 01 | 6-retriever metric comparison |
| `outputs/retrieval_paragraph_comparison.csv` | Notebook 01 | Paragraph-level metrics |
| `outputs/retrieval_bootstrap_ci.csv` | Notebook 01 | 95 % bootstrap confidence intervals |
| `outputs/retrieval_per_query_scores.json` | Notebook 01 | Per-query score arrays |
| `outputs/classifications.csv` | Notebook 02 | Gold-std proxy classifications |
| `outputs/classifications_full.csv` | Notebook 02 | Classifications + full chunk texts |
| `outputs/judge_results.csv` | Notebook 02 | Judge scores for gold-std proxies |
| `outputs/whitepaper_evaluation.csv` | Notebook 04 | Full whitepaper eval export |
| `outputs/whitepaper_judge.csv` | Notebook 04 | Judge scores for whitepaper |
| `outputs/whitepaper_retrieved_chunks.csv` | Notebook 04 | k chunks per whitepaper query |
| `outputs/gold_doc_eval_metrics.csv` | `scripts/run_gold_doc_eval.py` | Small gold-standard document-level retrieval metrics |
| `outputs/gold_doc_eval_metrics.json` | `scripts/run_gold_doc_eval.py` | Same metrics as JSON + run metadata |
| `outputs/mteb_legalbench_metrics.csv` | `scripts/run_mteb_legalbench_eval.py` | MTEB LegalBench retrieval metrics |
| `outputs/mteb_legalbench_metrics.json` | `scripts/run_mteb_legalbench_eval.py` | Same metrics as JSON + run metadata |
