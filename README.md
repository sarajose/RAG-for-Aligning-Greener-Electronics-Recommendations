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

**Compare classifiers** (Qwen2.5-7B vs Mistral-7B):

```python
from rag.classifier import AlignmentClassifier
clf_qwen    = AlignmentClassifier(model_key="qwen")    # default (7B)
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

**One-command local + Kaggle merge** (run local eval and merge downloaded Kaggle metrics in the same command):

```powershell
python main.py evaluate `
  --models bge-m3 e5-large-v2 e5-mistral `
  --output-dir outputs/eval_unified `
  --remote-eval-csv outputs/kaggle/e5_mistral_eval/eval_unified/metrics_all.csv
```

**Merge-only command** (if local evaluation was already run):

```powershell
python main.py merge-eval `
  --remote-csv outputs/kaggle/e5_mistral_eval/eval_unified/metrics_all.csv `
  --output-dir outputs/eval_unified
```

**Fast gold-standard-only** (no MTEB download):

```powershell
python main.py evaluate --models bge-m3 --skip-mteb
```

**Thesis study script** (Mar-17-style robust reproduction + ablation + baseline delta):

```powershell
python evaluation/full_study.py retrieval-study `
  --models bge-m3 e5-large-v2 e5-mistral `
  --include-splade `
  --with-robustness-all-models `
  --output-dir outputs/eval_thesis `
  --old-metrics-csv outputs/eval_unified_old/metrics_all.csv
```

**K comparison (k=1,3,5,10,20) from existing metrics CSV**:

```powershell
python evaluation/full_study.py k-compare `
  --metrics-csv outputs/eval_unified_old/metrics_all.csv `
  --k-values 1 3 5 10 20 `
  --output-dir outputs/eval_k_compare
```

**Prompt/judge analysis summary export**:

```powershell
python evaluation/full_study.py prompt-study `
  --prompt-csv outputs/prompt_results.csv `
  --judge-csv outputs/prompt_results_judge.csv `
  --output-dir outputs/eval_prompt
```

**Full pipeline (local + Kaggle merge for e5-mistral)**:

```powershell
# 1) Local evidence + indices (lightweight local models)
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m e5-large-v2
# Optional if your local machine can handle it:
# python main.py build -i outputs/evidence.csv -m e5-mistral

# 2) Local unified evaluation + robustness
python evaluation/full_study.py retrieval-study `
  --models bge-m3 e5-large-v2 `
  --include-splade `
  --with-robustness-all-models `
  --output-dir outputs/eval_thesis

# 3) Kaggle run for heavy model (e5-mistral-7b-instruct), then merge metrics locally
python main.py merge-eval `
  --remote-csv outputs/kaggle/e5_mistral_eval/eval_unified/metrics_all.csv `
  --output-dir outputs/eval_thesis

# 4) Optional Kaggle classifier + judge comparison
# Notebook: kaggle/kaggle_mistral_classifier_judge_pipeline.ipynb
# Upload: outputs/qwen_classifications.csv + outputs/qwen_classifications_retrieved_chunks.csv
# Download: classification_eval.csv, judge_scores_qwen3b.csv, judge_scores_mistral7b.csv
```

**Kaggle notebook handoff (explicit):**

1. Run `kaggle/kaggle_e5_mistral_eval_v2_pipeline.ipynb` to evaluate `e5-mistral` and export:
  - `eval_unified/metrics_all.csv` (for local `merge-eval`)
  - `retrieved_chunks.csv` (optional direct input for notebook 2)
2. Merge `metrics_all.csv` locally with `python main.py merge-eval ...`.
3. Select best retrieval model from merged metrics and run local prompt classification:
  - `python main.py prompt --model <BEST_MODEL> --output outputs/qwen_classifications.csv`
4. Run `kaggle/kaggle_mistral_classifier_judge_pipeline.ipynb` using:
  - `qwen_classifications.csv`
  - `retrieved_chunks.csv` or `qwen_classifications_retrieved_chunks.csv`
5. Use notebook 2 outputs in thesis tables/analysis:
  - `classification_eval.csv`, `judge_scores_qwen3b.csv`, `judge_scores_mistral7b.csv`

### 5. Pre-download models

```powershell
python main.py download-models --embedding-models bge-m3 e5-large-v2 e5-mistral
python main.py download-models --embedding-models bge-m3 --include-llms
```
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
| `--max-chunks-per-doc` | `2` | Cap chunks per document in split retrieval mode |
| `--near-dup-suppression` | off | Enable near-duplicate suppression in split mode |

### `evaluate`
| Argument | Default | Description |
|---|---|---|
| `--models` | bge-m3 e5-large-v2 e5-mistral | Model keys to compare |
| `--gold-csv` | `data/gold_standard_doc_level/gold_standard.csv` | Gold standard path |
| `--output-dir` | `outputs/eval_unified` | Output directory |
| `--top-k` | `10` | Retrieval candidates |
| `--rerank-top` | `5` | Results kept after reranking |
| `--export-k` | `10` | Number of retrieved chunks exported per query |
| `--k-values` | `1 3 5 10 20` | Evaluation cutoffs |
| `--whitepaper-csv` | recommendations CSV | Whitepaper recommendations path |
| `--skip-whitepaper` | off | Skip whitepaper chunk export |
| `--mteb-dataset` | `mteb/MuPLeR-retrieval` | MTEB retrieval dataset (English subset: `en-corpus`, `en-queries`, `en-qrels`) |
| `--mteb-split` | `test` | MTEB split |
| `--max-corpus` | `20000` | MTEB corpus cap |
| `--full-mteb` | off | Use full MTEB corpus |
| `--skip-mteb` | off | Skip MTEB legal tasks |
| `--skip-reranker` | off | Skip cross-encoder reranking |
| `--auto-build-indices` | off | Build missing indices automatically |
| `--evidence-csv` | `outputs/evidence.csv` | Evidence CSV used for auto-build |
| `--include-splade` | off | Include SPLADE sparse baseline |
| `--include-colbert` | off | Include BGE-M3 ColBERT multi-vector baseline (requires FlagEmbedding) |
| `--splade-model` | default in `config.py` | SPLADE model id |
| `--splade-max-length` | default in `config.py` | SPLADE max token length |
| `--remote-eval-csv` | none | Kaggle/remote `metrics_all.csv` path(s) to merge after local eval |
| `--force-cpu` | off | Disable GPU |
| `--with-robustness` | off | Run ablation significance tests |
| `--robust-model` | first model in `--models` | Model used for robustness stage |
| `--robust-k` | `10` | K used for robustness stage |
| `--rrf-k` | `60` | RRF smoothing constant for grid search (e.g. 10, 30, 60, 100) |

### `merge-eval`
| Argument | Default | Description |
|---|---|---|
| `--remote-csv` | required | One or more remote `metrics_all.csv` files |
| `--output-dir` | `outputs/eval_unified` | Unified output directory |
| `--ranking-k` | `10` | K used for ranking/summary regeneration |

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
| `outputs/eval_unified/gold_retrieved_chunks_<model>_<method>.csv` | Retrieved chunks for gold queries |
| `outputs/eval_unified/interpretation_k10.txt` | Auto-generated interpretation |
| `outputs/eval_unified/robustness/` | Bootstrap CI, permutation tests, ablation deltas |

---

## Evaluation design

Three complementary evaluation signals are produced in a single run:

| Signal | Level | Source |
|---|---|---|
| **Gold standard** | Document | 275 manual annotations (recommendation → EU regulation) |
| **Projected chunk** | Chunk (pseudo-relevance) | Same gold, all chunks from relevant docs marked relevant |
| **MTEB legal tasks** | Chunk | MuPLeR-retrieval (English subset): 10,000 EU legal docs, 200 queries |

Ablation configurations compared per model: `bm25`, `dense`, `rrf`, `bm25_rerank`, `dense_rerank`, `rrf_rerank`.

Core metrics: Hit@k, Recall@k, Precision@k, MRR, MAP, NDCG@k, Mean Rank, Chunk Hit Rate (ceiling proxy).

With `--with-robustness`: bootstrap 95% CI and paired permutation tests with significance stars.

---

## Models

**Embedding models** (key → HuggingFace ID):
- `bge-m3` → `BAAI/bge-m3`
- `e5-large-v2` → `intfloat/e5-large-v2`
- `e5-mistral` → `intfloat/e5-mistral-7b-instruct`

**LLM classifiers**:
- `qwen` → `Qwen/Qwen2.5-7B-Instruct` (default)
- `mistral` → `mistralai/Mistral-7B-Instruct-v0.3` (comparison baseline)

**Reranker**: `BAAI/bge-reranker-v2-m3` (multilingual, 570M parameters)
---

## File structure

```
main.py                          top-level CLI entry point
pipeline.py                      argparse CLI definitions
pipeline_commands.py             command implementations
pipeline_io.py                   I/O helpers (load/save CSV)
config.py                        all paths, model IDs, hyperparameters
data_models.py                   shared dataclasses (Chunk, Recommendation, ...)
embedding_indexing.py            embed_texts, build_faiss_index, load_indices (facade)

retrieval/
  retrieval.py                   HybridRetriever (BM25 + FAISS + RRF + reranker)
  hybrid_retriever.py            lightweight HybridRetriever for ablation evaluation
  bm25_retriever.py              BM25Retriever baseline
  dense_retriever.py             DenseRetriever baseline
  reranker.py                    Reranker, RerankedRetriever (cross-encoder)
  splade_retriever.py            SPLADE sparse baseline
  colbert_retriever.py           ColBERT multi-vector baseline
  base_retriever.py              BaseRetriever interface
  chunking_evidence.py           EUR-Lex HTML → structured CSV chunks
  chunking_recommendations.py    Recommendation CSV loader

indexing/
  embeddings.py                  embed_texts, get_embed_model
  indices.py                     build_faiss_index, build_bm25_index, load_indices
  chunks.py                      load_chunks, load_and_merge_chunks

evaluation/
  experiment_unified.py          unified evaluation orchestrator (gold + MTEB + ablation)
  experiment_helpers.py          shared stats/metrics/retriever-building helpers
  experiment_mteb.py             MTEB dataset loading and chunk-level evaluation
  experiment_baselines.py        SPLADE and ColBERT baseline evaluation helpers
  experiment_robustness.py       robustness analysis (bootstrap CI, permutation tests)
  experiment_exports.py          chunk export helpers
  experiment_commands.py         thin CLI entrypoints (merge-eval, download-models)
  full_study.py                  thesis full-study CLI (retrieval-study, prompt-study, k-compare)
  evaluation.py                  core evaluation logic (gold standard loader, per-query scoring)
  full_eval.py                   ablation table, significance markers, report formatting
  metrics.py                     Hit@k, Recall, MRR, NDCG, bootstrap CI, permutation test

rag/
  classifier.py                  AlignmentClassifier (Qwen / Mistral)
  llm_judge.py                   LLMJudge (LLM-as-judge evaluation)
  prompts.py                     prompt templates

notebooks/                       analysis and visualisation
outputs/                         generated artifacts (indices, results, eval)
data/                            evidence HTML, gold standard, recommendations
docs/                            CLI reference and pipeline walkthroughs
```
