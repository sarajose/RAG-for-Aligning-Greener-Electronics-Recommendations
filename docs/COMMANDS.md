# Commands Reference — Unified Thesis Evaluation Pipeline

This workflow keeps evaluation in Python scripts and visualization in one notebook.

## 0) Activate Environment

```powershell
.\venv\Scripts\Activate.ps1
```

## 1) Install Dependencies

```powershell
pip install -r requirements.txt
```

## 2) (Optional) Pre-download Models

This avoids repeated Hugging Face downloads during evaluation runs.

```powershell
python pipeline.py download-models
```

Optional (large download):

```powershell
python pipeline.py download-models --include-llms
```

## 3) Build Retrieval Indices (per embedding model)

Run once per model key you want to compare:

```powershell
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m mpnet
python main.py build -i outputs/evidence.csv -m minilm
```

Index outputs are stored in `outputs/indices/`.

## 4) Run Unified Evaluation

This single command runs:
- Gold-standard document-level retrieval evaluation (multiple methods).
- MTEB LegalBench chunk-level evaluation (model comparison).
- Top-k retrieved chunk export for gold-standard queries.
- Top-k retrieved chunk export for whitepaper recommendations.

```powershell
python pipeline.py unified-eval `
    --models bge-m3 mpnet minilm `
    --k-values 1 3 5 10 20 `
    --top-k 20 `
    --rerank-top 10 `
    --export-k 10
```

Full-corpus MTEB run (no corpus cap):

```powershell
python pipeline.py unified-eval `
    --models bge-m3 mpnet minilm `
    --k-values 1 3 5 10 20 `
    --top-k 20 `
    --rerank-top 10 `
    --export-k 10 `
    --full-mteb
```

Faster test run:

```powershell
python pipeline.py unified-eval `
    --models bge-m3 `
    --max-corpus 10000
```

Low-memory baseline run (CPU + automatic index build):

```powershell
python pipeline.py unified-eval `
    --models minilm `
    --skip-reranker `
    --force-cpu `
    --max-corpus 2000 `
    --auto-build-indices
```

## 5) Visualize Results

Use the single notebook:

- `notebooks/05_eval_visualizations_only.ipynb`

It reads outputs from `outputs/eval_unified/` and plots:
- Gold document-level method/model comparison.
- MTEB chunk-level comparison.
- Whitepaper retrieval inspection tables/charts.

## Main Output Files

All unified outputs are saved in `outputs/eval_unified/`.

| File | Description |
|------|-------------|
| `metrics_all.csv` | All metrics across datasets/methods/models/k |
| `ranking_k10.csv` | Publication-friendly ranking at k=10 |
| `run_summary.json` | Run metadata and output pointers |
| `gold_retrieved_chunks_<model>_<method>.csv` | Top-k chunks per gold query |
| `mteb_retrieved_chunks_<model>_<method>.csv` | Top-k chunks per MTEB query |
| `whitepaper_retrieved_chunks_<model>_<method>.csv` | Top-k chunks per whitepaper recommendation |

## Methodology Mapping

- Retrieval-first, then interpretation.
- In-domain benchmark: gold-standard document links.
- External benchmark context: MTEB LegalBench chunk retrieval.
- Application stage: whitepaper recommendations with exported retrieved evidence for review.

## 6) Whitepaper LLM Classification + Judge

When new recommendations are added to the whitepaper CSV, run:

```powershell
python pipeline.py whitepaper `
    --input data/recommendations_whitepaper/recommendations_empty.csv `
    --model bge-m3 `
    --top-k 10 `
    --rerank-top 5 `
    --judge
```

Low-VRAM mode:

```powershell
python pipeline.py whitepaper `
    --input data/recommendations_whitepaper/recommendations_empty.csv `
    --model minilm `
    --no-rerank `
    --judge
```

Outputs are written to `outputs/whitepaper_classified.csv` and companion retrieved/judge CSV files.

## 7) Robustness and Publishability Exports

```powershell
python pipeline.py robustness `
    --model bge-m3 `
    --k 10
```

This generates CI tables, pairwise permutation tests, ablation deltas,
error categories, and hard-negative cases in `outputs/eval_unified/robustness/`.

## 8) Reproducibility Manifest

No dedicated manifest script is included in this repository. For reproducibility,
store the exact command, git commit hash, and generated files under `outputs/eval_unified/`.

## 9) Human Agreement (Two Raters)

After evaluators complete `evaluator_1_label` and `evaluator_2_label` in
`outputs/whitepaper_classified.csv`, compute Cohen's kappa in a notebook or stats tool.



