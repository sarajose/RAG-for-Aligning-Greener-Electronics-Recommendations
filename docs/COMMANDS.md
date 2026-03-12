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
| 4 | Generate benchmark JSON | <1 min |
| 5 | Evaluate retrieval — benchmark | ~2–5 min |
| 6 | Classify whitepaper recommendations | ~2–4 h (GPU) |
| 7 | LLM-as-judge on classifications | ~1–2 h (GPU) |
| 8 | Full pipeline (steps 3+6+7 combined) | ~3–6 h (GPU) |

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

## Step 4 — Generate Internal Benchmark JSON

Converts the gold standard CSV into a self-contained JSON benchmark file
that can be shared or used for reproducible evaluation runs.

```powershell
python benchmarks/generate_benchmark.py `
    --gold data/gold_standard_doc_level/gold_standard.csv `
    -o benchmarks/gold_standard_benchmark.json
```

**Output**: `benchmarks/gold_standard_benchmark.json`

> **Do you need an external benchmark?**  
> The gold standard benchmark above is derived from your own annotations and
> is sufficient for this task.  If you want to validate on a truly external
> dataset, the closest publicly available option is the
> [BEIR/LegalBench](https://huggingface.co/datasets/BeIR/arguana) suite or
> the [GDPR-QA](https://huggingface.co/datasets/joelito/gdpr_qa) dataset on
> Hugging Face. Download one of those and place it in `benchmarks/` with the
> format `[{"query": "...", "relevant_ids": ["chunk_id_1", ...]}, ...]`.
> **No external download is strictly required** for the main pipeline.

---

## Step 5 — Evaluate Retrieval on Benchmark

```powershell
python main.py benchmark `
    -i benchmarks/gold_standard_benchmark.json `
    -o outputs/metrics_benchmark.json
```

**Outputs**:
- Console: formatted retrieval report for the benchmark
- `outputs/metrics_benchmark.json`

---

## Step 6 — Classify Whitepaper Recommendations

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

## Step 7 — LLM-as-Judge on Classifications

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

## Step 8 — Full Pipeline (Retrieval → Classification → Judge)

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

## Notebook Workflow

After any pipeline run, open the notebooks for analysis and visualisation.
Results are cached in `outputs/` so **no re-computation happens** unless you
set `FORCE_RERUN = True` at the top of each notebook.

| Notebook | Purpose |
|----------|---------|
| `notebooks/01_retrieval_analysis.ipynb` | Compare 6 retriever configs, bootstrap CIs, permutation tests |
| `notebooks/02_rag_evaluation.ipynb` | Classification + LLM-as-judge on gold-standard queries |
| `notebooks/03_evaluation_metrics.ipynb` | Consolidated metrics dashboard |
| `notebooks/04_whitepaper_evaluation.ipynb` | Whitepaper recommendations analysis + human eval export |

---

## Quick-Reference: All Output Files

| File | Generated by | Description |
|------|-------------|-------------|
| `outputs/evidence.csv` | chunking_evidence.py | All EU legislation chunks |
| `outputs/indices/bge-m3_*.index/.pkl` | `main.py build` | FAISS + BM25 indices |
| `outputs/metrics_retrieval.json` | `main.py evaluate` | Retrieval metrics (gold std) |
| `outputs/metrics_benchmark.json` | `main.py benchmark` | Retrieval metrics (benchmark) |
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
