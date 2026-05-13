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


```powershell
python main.py merge-eval --remote-csv outputs/eval_mistral/metrics_all.csv outputs/eval_mteb_k_split/metrics_all.csv --output-dir outputs/eval_mteb_k_split_with_mistral
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

**Full pipeline (evidence → indices → evaluation):**

```powershell
# 1) Chunk legal evidence
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv

# 2) Build indices for each embedding model
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m e5-large-v2
python main.py build -i outputs/evidence.csv -m e5-mistral

# 3) Run unified evaluation + robustness analysis
python evaluation/full_study.py retrieval-study `
  --models bge-m3 e5-large-v2 e5-mistral `
  --include-splade `
  --with-robustness-all-models `
  --output-dir outputs/eval_thesis
```

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
| `--splade-model` | default in `config.py` | SPLADE model id |
| `--splade-max-length` | default in `config.py` | SPLADE max token length |
| `--force-cpu` | off | Disable GPU |
| `--with-robustness` | off | Run ablation significance tests |
| `--robust-model` | first model in `--models` | Model used for robustness stage |
| `--robust-k` | `10` | K used for robustness stage |
| `--rrf-k` | `60` | RRF smoothing constant for grid search (e.g. 10, 30, 60, 100) |

---

## Inputs

| Path | Description |
|---|---|
| `data/evidence/` | EUR-Lex HTML files |
| `data/gold_standard_doc_level/gold_standard.csv` | 275 document-level annotations |
| `data/recommendations_whitepaper/recommendations_v2.csv` | Whitepaper recommendations |
| `outputs/evidence.csv` | Generated chunk file |

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
| `outputs/eval_unified/robustness/` | Bootstrap CI, permutation tests, ablation deltas (not used)|

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
  experiment_baselines.py        SPLADE baseline evaluation helpers
  experiment_robustness.py       robustness analysis (bootstrap CI, permutation tests)
  experiment_exports.py          chunk export helpers
  experiment_commands.py         thin CLI entrypoints (download-models)
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
