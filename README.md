
# RAG-for-Aligning-Greener-Electronics-Recommendations

This repository implements a reproducible RAG pipeline to map sustainability recommendations to EU legislation, then evaluate retrieval quality across in-domain and external legal benchmarks.

## Pipeline Overview

The pipeline is intentionally modular and runs in this order:

1. Evidence chunking
	- Script: `retrieval/chunking_evidence.py`
	- Input: HTML files in `data/evidence/`
	- Output: chunk CSV (typically `outputs/evidence.csv`)

2. Optional recommendation chunking
	- Script: `retrieval/chunking_recommendations.py`
	- Input: structured TXT recommendations
	- Output: recommendation CSV

3. Embedding + indexing
	- Entry point: `python main.py build ...`
	- Internal modules: `embedding_indexing.py`, `indexing/`
	- Output per model: FAISS + BM25 + chunk artifacts in `OUTPUT_DIR/indices/`

4. Retrieval + optional classification/judge
	- Entry point: `python main.py prompt ...`
	- Internal modules: `pipeline_commands.py`, `pipeline_io.py`, `retrieval/`, `rag/`
	- Outputs: prompt results CSV, retrieved chunks CSV, optional judge CSV

5. Unified retrieval evaluation + optional robustness stats
	- Entry point: `python main.py evaluate ...`
	- Internal modules: `evaluation/experiment_commands.py`, `evaluation/evaluation.py`, `evaluation/metrics.py`
	- Outputs: metrics tables, rankings, retrieved chunks exports, robustness/significance tables

6. Visualization
	- Notebook: `notebooks/05_eval_visualizations_only.ipynb`
	- Reads the configured output directory from `config.py`

Primary CLI entry points:

- `main.py` (recommended top-level CLI)
- `pipeline.py` (CLI parser/dispatcher used by `main.py`)
- `embedding_indexing.py` (legacy indexing CLI compatibility)

## Execution Order

Use these steps in sequence for a clean run.

### 0. Environment setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --help
```

### 1. Chunk evidence (required when corpus changes)

```powershell
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv
```

Depends on:
- HTML files in `data/evidence/`

Produces:
- `outputs/evidence.csv`

### 2. Build indices (required before prompt/evaluate unless auto-build is used)

```powershell
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m mpnet
python main.py build -i outputs/evidence.csv -m minilm
```

Depends on:
- evidence CSV from step 1

Produces (per model key):
- `OUTPUT_DIR/indices/<model>_faiss.index`
- `OUTPUT_DIR/indices/<model>_bm25.pkl`
- `OUTPUT_DIR/indices/<model>_chunks.pkl`

### 3. Prompt pipeline (optional; for whitepaper/applications)

```powershell
python main.py prompt `
	--input data/recommendations_whitepaper/recommendations_empty.csv `
	--output outputs/prompt_results.csv `
	--model bge-m3 `
	--top-k 10 `
	--rerank-top 5 `
  --judge
```

Depends on:
- recommendation CSV
- built indices for selected model

Produces:
- `outputs/prompt_results.csv`
- `outputs/prompt_results_retrieved_chunks.csv`
- `outputs/prompt_results_judge.csv` (if `--judge`)

### 4. Unified evaluation (required for benchmark tables)

```powershell
python main.py evaluate `
	--models bge-m3 mpnet minilm `
	--include-splade `
	--k-values 1 3 5 10 20 `
	--top-k 20 `
	--rerank-top 10 `
	--export-k 10 `
  --with-robustness
```

Depends on:
- gold labels: `data/gold_standard_doc_level/gold_standard.csv`
- whitepaper recommendations (unless `--skip-whitepaper`)
- model indices (or `--auto-build-indices` + evidence CSV)

Produces under `--output-dir` (default: `OUTPUT_DIR/eval_unified`):
- `metrics_all.csv`
- `ranking_k10.csv`
- `run_summary.json`
- `gold_retrieved_chunks_<model>_<method>.csv`
- `mteb_retrieved_chunks_<model>_<method>.csv`
- `whitepaper_retrieved_chunks_<model>_<method>.csv`
- `robustness/*.csv` (if `--with-robustness`)

### 5. Notebook visualization (optional)

Open:
- `notebooks/05_eval_visualizations_only.ipynb`

The notebook expects outputs from step 4.

## Outputs

### Where outputs are saved

All output paths are controlled centrally in `config.py`:

- `OUTPUT_DIR`
- `INDEX_DIR = OUTPUT_DIR/indices`

Key conventions:

1. Index artifacts are model-scoped by filename prefix.
	- Example: `<model>_faiss.index`, `<model>_bm25.pkl`, `<model>_chunks.pkl`

2. Retrieval export filenames include model and method.
	- Example: `gold_retrieved_chunks_bge-m3_rrf_rerank.csv`

3. Unified metrics are aggregated in fixed schema files.
	- `metrics_all.csv`
	- `ranking_k10.csv`
	- `metrics_summary_k10.csv`
	- `comparison_k10.csv`
	- `interpretation_k10.txt`
	- `run_summary.json`

4. Robustness outputs are separated under `robustness/`.
	- CI tables, pairwise permutation p-values, ablation deltas, error analysis

### Reuse across stages

1. `outputs/evidence.csv` (or your chosen evidence CSV) feeds index building.
2. Index files in `OUTPUT_DIR/indices/` are reused by prompt and evaluation.
3. Unified evaluation outputs are reused by the notebook.

## Evaluation Logic

This section describes what is scored and why comparisons are fair.

1. Gold document-level retrieval
	- Loaded via `evaluation/evaluation.py`.
	- Gold mappings are grouped as query -> set of canonical document names.
	- Retrieved chunks are deduplicated to document ranking before metrics.

2. MTEB LegalBench chunk-level retrieval
	- Built from dataset corpus/query/qrels splits.
	- Relevance is based on qrels overlap with available corpus IDs.

3. Metrics
	- Computed by `evaluation/metrics.py`:
	  - Hit@k, Recall@k, Precision@k, MRR, MAP, NDCG, Mean Rank.

4. Statistical comparison (robustness mode)
	- Per-query scores are produced per method on the same query order.
	- 95% bootstrap CIs (10,000 resamples, fixed seed).
	- Paired permutation tests (10,000 permutations, fixed seed) are run on aligned per-query vectors.
	- Holm-Bonferroni correction is applied across pairwise tests per metric.
	- Practical effect size (`Cohen's dz`) is reported with each pairwise comparison.

5. Fairness assumptions
	- Methods compared for the same model share the same index/chunk artifacts.
	- Cross-model fairness requires that all model indices were built from the same evidence CSV/chunking setup.
	- The code does not enforce evidence-hash equality, so keep build inputs identical across models.

## Common Failure Points

1. Indices missing before prompt
	- Symptom: prompt fails before retrieval.
	- Fix: run build first for the selected model.

2. Output directory confusion
	- Symptom: notebook cannot find `metrics_all.csv`.
	- Fix: verify `OUTPUT_DIR` in `config.py` and run evaluate again (or pass `--output-dir`).

3. Whitepaper file missing
	- Symptom: evaluate fails unless `--skip-whitepaper`.
	- Fix: provide file or skip whitepaper export.

4. Inconsistent model comparisons
	- Symptom: unfair/unstable cross-model results.
	- Fix: rebuild all model indices from the same evidence CSV.

5. Large-model resource pressure
	- Symptom: LLM classifier/judge or reranker OOM/slow runs.
	- Fix: run retrieval-only, skip reranker, reduce evaluated models, or use CPU flags.

## Additional References

- `docs/COMMANDS.md` for command-focused run snippets.
- `docs/PIPELINE.md` for detailed method walkthrough.
- `docs/METHODOLOGY.md` for thesis-oriented methodological detail.