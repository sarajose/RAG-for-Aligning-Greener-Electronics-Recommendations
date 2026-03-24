# RAG-for-Aligning-Greener-Electronics-Recommendations

## Project

Reproducible Retrieval-Augmented Generation (RAG) workflow that links sustainability recommendations to EU legislation and evaluates retrieval quality.

Pipeline stages:
1. Chunk legal evidence (EUR-Lex HTML → CSV).
2. Build retrieval indices (embedding + BM25).
3. Retrieve evidence and classify alignment (optionally with LLM judge).
4. Run unified evaluation: document-level, projected chunk-level, and MTEB legal tasks.
5. Visualise results in the notebook.

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

### 1. Chunk evidence documents

Parse EUR-Lex HTML files into a structured CSV of legal provisions:

```powershell
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv
```

### 2. Build indices

Build FAISS + BM25 indices for each embedding model you want to compare:

```powershell
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m e5-large-v2
python main.py build -i outputs/evidence.csv -m e5-mistral
```

### 3. Retrieve and classify

**Flat baseline** (BM25 + dense + RRF + reranker):

```powershell
python main.py prompt `
  --input data/recommendations_whitepaper/recommendations_v2.csv `
  --output outputs/prompt_results.csv `
  --model bge-m3 `
  --top-k 10 `
  --rerank-top 5 `
  --judge
```

**Split evidence retrieval** (binding law vs. policy docs retrieved separately):

```powershell
python main.py prompt `
  --input data/recommendations_whitepaper/recommendations_v2.csv `
  --output outputs/prompt_results_split.csv `
  --model bge-m3 `
  --retrieval-mode split_evidence_retrieval `
  --judge
```

**Retrieve only** (skip LLM classification):

```powershell
python main.py prompt --retrieve-only --model bge-m3
```

**Compare classifiers** (Qwen2.5-3B vs Mistral-7B):

```python
from rag.classifier import AlignmentClassifier
clf_qwen    = AlignmentClassifier(model_key="qwen")    # default
clf_mistral = AlignmentClassifier(model_key="mistral") # for comparison
```

### 4. Evaluate

**Full unified evaluation** (document-level gold + projected chunk-level + MTEB legal suite + ablation table):

```powershell
python main.py evaluate `
  --models bge-m3 e5-large-v2 e5-mistral `
  --k-values 1 3 5 10 20 `
  --top-k 10
```

**Ablation with significance stars** (adds per-query scoring + permutation-test markers):

```powershell
python main.py evaluate `
  --models bge-m3 e5-large-v2 e5-mistral `
  --k-values 1 3 5 10 20 `
  --top-k 10 `
  --with-robustness
```

**Fast gold-standard-only** (no MTEB download):

```powershell
python main.py evaluate --models bge-m3 --skip-mteb
```

**Standalone evaluation script** (same as above, no main pipeline required):

```powershell
python evaluation/run_eval.py --models bge-m3 --skip-mteb --output-dir results/eval
python evaluation/run_eval.py --models bge-m3 e5-large-v2 --output-dir results/eval
python evaluation/run_eval.py --models bge-m3 --with-robustness --output-dir results/eval
```

### 5. Pre-download models

```powershell
python main.py download-models --embedding-models bge-m3 e5-large-v2 e5-mistral
python main.py download-models --embedding-models bge-m3 --include-llms
```

### 6. Open visualisation notebook

`notebooks/05_eval_visualizations_only.ipynb`

---

## CLI reference

### `build`
| Argument | Default | Description |
|---|---|---|
| `-i / --input` | required | Evidence CSV file(s) |
| `-m / --model` | `bge-m3` | Embedding model key |

### `prompt`
| Argument | Default | Description |
|---|---|---|
| `-i / --input` | whitepaper CSV | Recommendations CSV |
| `-o / --output` | `outputs/prompt_results.csv` | Output CSV |
| `-m / --model` | `bge-m3` | Embedding model key |
| `-k / --top-k` | `10` | Candidates before reranking |
| `--rerank-top` | `5` | Results after reranking |
| `--retrieval-mode` | `flat_baseline` | `flat_baseline` or `split_evidence_retrieval` |
| `--no-rerank` | off | Skip cross-encoder reranking |
| `--retrieve-only` | off | Skip LLM classification |
| `--judge` | off | Run LLM judge after classification |

### `evaluate`
| Argument | Default | Description |
|---|---|---|
| `--models` | bge-m3 e5-large-v2 e5-mistral | Model keys to compare |
| `--gold-csv` | `data/gold_standard_doc_level/gold_standard.csv` | Gold standard path |
| `--output-dir` | `outputs/eval_unified` | Output directory |
| `--top-k` | `10` | Retrieval candidates |
| `--k-values` | `1 3 5 10 20` | Evaluation cutoffs |
| `--skip-mteb` | off | Skip MTEB legal tasks |
| `--skip-reranker` | off | Skip cross-encoder reranking |
| `--force-cpu` | off | Disable GPU |
| `--with-robustness` | off | Run ablation significance tests |

---

## Inputs

| Path | Description |
|---|---|
| `data/evidence/` | EUR-Lex HTML files |
| `data/gold_standard_doc_level/gold_standard.csv` | 275 document-level annotations |
| `data/recommendations_whitepaper/recommendations_v2.csv` | Whitepaper recommendations |
| `outputs/evidence.csv` | Generated chunk file (from step 1) |

## Outputs

| Path | Description |
|---|---|
| `outputs/indices/` | FAISS + BM25 index artifacts per model |
| `outputs/prompt_results.csv` | Classification results |
| `outputs/prompt_results_retrieved_chunks.csv` | Retrieved evidence per recommendation |
| `outputs/eval_unified/metrics_all.csv` | All metrics across models/methods/k |
| `outputs/eval_unified/ablation_table.csv` | Ablation table (method × model × metric) |
| `outputs/eval_unified/ablation_table.txt` | Human-readable ablation report |
| `outputs/eval_unified/per_query_scores_for_ablation.csv` | Per-query scores used for significance stars (`--with-robustness`) |
| `outputs/eval_unified/ranking_k10.csv` | Models ranked by NDCG@10 |
| `outputs/eval_unified/metrics_summary_k10.csv` | Summary table at k=10 |
| `outputs/eval_unified/comparison_k10.csv` | Best vs second model gaps |
| `outputs/eval_unified/gold_retrieved_chunks_<model>.csv` | Retrieved chunks for gold queries |
| `outputs/eval_unified/interpretation_k10.txt` | Auto-generated interpretation |
| `outputs/eval_unified/robustness/` | Bootstrap CI, permutation tests, ablation deltas |

---

## Evaluation design

Three complementary evaluation signals are produced in a single run:

| Signal | Level | Source |
|---|---|---|
| **Gold standard** | Document | 275 manual annotations (recommendation → EU regulation) |
| **Projected chunk** | Chunk (pseudo-relevance) | Same gold, all chunks from relevant docs marked relevant |
| **MTEB legal tasks** | Chunk | MTEB (eng, v2) Retrieval tasks filtered to legal domain |

Ablation configurations compared per model: `bm25`, `dense`, `rrf`, `bm25_rerank`, `dense_rerank`, `rrf_rerank`.

Core metrics: Hit@k, Recall@k, Precision@k, MRR, MAP, NDCG@k, Mean Rank.

With `--with-robustness`: bootstrap 95% CI, paired permutation tests with Holm-Bonferroni correction, and per-ablation delta table.

---

## Models

**Embedding models** (key → HuggingFace ID):
- `bge-m3` → `BAAI/bge-m3`
- `e5-large-v2` → `intfloat/e5-large-v2`
- `e5-mistral` → `intfloat/e5-mistral-7b-instruct`
- `mpnet` → `sentence-transformers/all-mpnet-base-v2`

**LLM classifiers**:
- `qwen` → `Qwen/Qwen2.5-3B-Instruct` (default)
- `mistral` → `mistralai/Mistral-7B-Instruct-v0.3` (comparison baseline)

**Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## File structure

```
main.py                        top-level CLI entry point
pipeline.py                    argparse CLI definitions
pipeline_commands.py           command implementations
pipeline_io.py                 I/O helpers (load/save CSV)
config.py                      all paths, model IDs, hyperparameters
data_models.py                 shared dataclasses (Chunk, Recommendation, ...)

retrieval/
  retrieval.py                 HybridRetriever (BM25 + FAISS + RRF + reranker)
                               + BM25Retriever, DenseRetriever for ablation
  reranker.py                  Reranker, RerankedRetriever (cross-encoder)
  chunking_evidence.py         EUR-Lex HTML → structured CSV chunks
  chunking_recommendations.py  Recommendation CSV loader

indexing/
  embeddings.py                embed_texts, get_embed_model
  indices.py                   build_faiss_index, build_bm25_index, load_indices
  chunks.py                    load_chunks, load_and_merge_chunks

evaluation/
  evaluate.py                  unified evaluation (gold + projected + MTEB + ablation)
  run_eval.py                  standalone evaluation CLI
  metrics.py                   Hit@k, Recall, MRR, NDCG, bootstrap CI, ...

rag/
  classifier.py                AlignmentClassifier (Qwen / Mistral)
  llm_judge.py                 LLMJudge (LLM-as-judge evaluation)
  prompts.py                   prompt templates

notebooks/                     analysis and visualisation
outputs/                       generated artifacts (indices, results, eval)
data/                          evidence HTML, gold standard, recommendations
```
