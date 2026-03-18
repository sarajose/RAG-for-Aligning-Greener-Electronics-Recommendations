# RAG-for-Aligning-Greener-Electronics-Recommendations

## Project

This project builds a reproducible Retrieval-Augmented Generation (RAG) workflow that links sustainability recommendations to EU legislation and evaluates retrieval quality.

Main stages:
1. Chunk legal evidence.
2. Build retrieval indices.
3. Run retrieval (optionally with classification and judge).
4. Run unified evaluation and robustness analysis.
5. Visualize results in the notebook.

## Usage

### Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Build data and indices

```powershell
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m mpnet
python main.py build -i outputs/evidence.csv -m minilm
python main.py build -i outputs/evidence.csv -m e5-large-v2
```

### Run unified evaluation

```powershell
python main.py evaluate `
  --models bge-m3 mpnet minilm e5-large-v2 `
  --include-splade `
  --k-values 1 3 5 10 20 `
  --top-k 20 `
  --rerank-top 10 `
  --export-k 10 `
  --with-robustness
```

### Open visualization notebook

- notebooks/05_eval_visualizations_only.ipynb

## Inputs and Outputs

### Inputs

- data/evidence/ (EU legal HTML files)
- data/gold_standard_doc_level/gold_standard.csv
- data/recommendations_whitepaper/recommendations_empty.csv (optional for whitepaper export)
- outputs/evidence.csv (generated chunk file)

### Outputs

Under outputs/eval_unified/:
- metrics_all.csv
- ranking_k10.csv
- metrics_summary_k10.csv
- comparison_k10.csv
- run_summary.json
- gold_retrieved_chunks_<model>_<method>.csv
- mteb_retrieved_chunks_<model>_<method>.csv
- whitepaper_retrieved_chunks_<model>_<method>.csv
- robustness/*.csv (if enabled)

Index artifacts are saved under outputs/indices/.

## Models

Common embedding model keys:
- bge-m3
- mpnet
- minilm
- e5-large-v2

Sparse baseline:
- splade (enabled with --include-splade)

Methods compared in evaluation include bm25, dense, rrf, and reranked variants.

## File Structure

- main.py: top-level CLI.
- retrieval/: chunking and retriever implementations.
- indexing/: embedding and index utilities.
- evaluation/: metrics, experiment runner, robustness statistics.
- rag/: prompt, classifier, judge modules.
- docs/: methodology and command docs.
- notebooks/: analysis and plots.
- outputs/: generated artifacts.

## Evaluation

Evaluation combines:
1. Gold-standard document-level retrieval.
2. MTEB LegalBench chunk-level retrieval.
3. Whitepaper retrieval.

Core metrics: Hit@k, Recall@k, Precision@k, MRR, MAP, NDCG, Mean Rank.

With robustness enabled, the pipeline also reports bootstrap confidence intervals, paired permutation tests.