# RAG-for-Aligning-Greener-Electronics-Recommendations

## Project

Reproducible Retrieval-Augmented Generation (RAG) workflow that links sustainability recommendations to EU legislation and evaluates retrieval quality.

Pipeline stages:
1. Chunk legal evidence (EUR-Lex HTML to CSV).
2. Chunk recommendations whitepaper to CSV.
3. Build retrieval indices (embedding and BM25).
4. Retrieve evidence and classify alignment.
5. Run unified evaluation

---

## Usage

### 1. Chunk evidence documents

Parse EUR-Lex HTML files into a structured CSV of legal provisions:

```powershell
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv
```

### 2. Chunk recommendations

Parse the whitepaper recommendations text into a structured CSV:

```powershell
python retrieval/chunking_recommendations.py -i data/recommendations_whitepaper/recommendations.txt -o data/recommendations_whitepaper/recommendations_v2.csv
```

### 3. Build indices

Build FAISS + BM25 indices for each embedding model you want to compare:

```powershell
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m e5-large-v2
python main.py build -i outputs/evidence.csv -m e5-mistral
```

### 4. Retrieve and classify

**Split evidence retrieval** (binding law vs. policy docs retrieved separately):
(remove retrieval mode for flat baseline)

```powershell
python main.py prompt `
  --input data/recommendations_whitepaper/recommendations_v2.csv `
  --output outputs/prompt_results_split.csv `
  --model bge-m3 `
  --retrieval-mode split_evidence_retrieval `
  --judge
```

### 5. Evaluate

**Full unified evaluation** (document-level gold + MTEB legal suite + SPLADE baseline):

```powershell
python main.py evaluate `
  --models bge-m3 e5-large-v2 `
  --k-values 1 3 5 10 20 `
  --top-k 10
```

If some models have been evaluated separately, merge their results:

```powershell
python main.py merge-eval `
  --remote-csv outputs/eval_mistral/metrics_all.csv `
  --output-dir outputs/eval_unified
```

**Prompt/judge analysis summary export**:

```powershell
python evaluation/full_study.py prompt-study `
  --prompt-csv outputs/prompt_results.csv `
  --judge-csv outputs/prompt_results_judge.csv `
  --output-dir outputs/eval_prompt
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
| `--inner-retrieval-method` | `rrf` | Inner fusion method in split mode (`rrf`, `dense`, `bm25`) |
| `--judge` | off | Run LLM judge after classification |

### `evaluate`
| Argument | Default | Description |
|---|---|---|
| `--models` | bge-m3 e5-large-v2 e5-mistral | Model keys to compare |
| `--output-dir` | `outputs/eval_unified` | Output directory |
| `--top-k` | `10` | Retrieval candidates |
| `--rerank-top` | `5` | Results kept after reranking |
| `--k-values` | `1 3 5 10 20` | Evaluation cutoffs |
| `--retrieval-mode` | `flat_baseline` | `flat_baseline` or `split_evidence_retrieval` |
| `--evidence-csv` | `outputs/evidence.csv` | Evidence CSV for auto-building missing indices |

Always on (not configurable): SPLADE baseline, full MTEB corpus (MuPLeR, no cap), cross-encoder reranking, auto-build missing indices.

### `merge-eval`
| Argument | Default | Description |
|---|---|---|
| `--remote-csv` | required | One or more metrics CSVs to merge |
| `--output-dir` | `outputs/eval_unified` | Output directory |

---

## Inputs

| Path | Description |
|---|---|
| `data/evidence/` | EUR-Lex HTML files |
| `data/gold_standard_doc_level/gold_standard.csv` | 130 document-level annotations (private) |
| `data/recommendations_whitepaper/recommendations_v2.csv` | Whitepaper recommendations (private) |
| `outputs/evidence.csv` | Generated chunk file |

## Outputs (not public yet as part of the dataset is private)

| Path | Description |
|---|---|
| `outputs/indices/` | FAISS + BM25 index artifacts per model |
| `outputs/prompt_results.csv` | Classification results |
| `outputs/prompt_results_retrieved_chunks.csv` | Retrieved evidence per recommendation |
| `outputs/eval_unified/metrics_all.csv` | All metrics across models/methods/k (used by retrieval notebook) |
| `outputs/eval_unified/mteb_retrieved_chunks_<model>_<method>.csv` | Retrieved MTEB chunks per configuration |
| `outputs/eval_prompt/classification_label_distribution.csv` | Label distribution (used by classifier notebook) |
| `outputs/eval_prompt/classification_retrieval_mode_distribution.csv` | Retrieval mode distribution |
| `outputs/eval_prompt/classification_cited_chunk_frequency.csv` | Most-cited chunk IDs |
| `outputs/eval_prompt/judge_overall_band_distribution.csv` | Judge score band distribution (used by judge notebook) |

---

## Models

**Embedding models**:
- `bge-m3` → `BAAI/bge-m3`
- `e5-large-v2` → `intfloat/e5-large-v2`
- `e5-mistral` → `intfloat/e5-mistral-7b-instruct`

**LLM classifiers and judge**:
- `mistral` → `mistralai/Mistral-7B-Instruct-v0.3` (classifier)
- `qwen` → `Qwen/Qwen2.5-7B-Instruct` (judge)

**Reranker**: `BAAI/bge-reranker-v2-m3` (multilingual, 570M parameters)

**Sparse baseline**: SPLADE (always included in evaluation)

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
  chunking_recommendations.py    whitepaper recommendations TXT → structured CSV

indexing/
  embeddings.py                  embed_texts, get_embed_model
  indices.py                     build_faiss_index, build_bm25_index, load_indices
  chunks.py                      load_chunks, load_and_merge_chunks

evaluation/
  retrieval_eval.py              unified evaluation orchestrator (gold + MTEB + SPLADE)
  mteb_eval.py                   MTEB dataset loading and chunk-level evaluation
  commands.py                    CLI entrypoints (merge-eval, download-models)
  full_study.py                  prompt-study CLI (classification + judge analysis)
  generate_judge_from_classifications.py  re-run LLM judge on existing classification CSV
  evaluation.py                  core metrics and gold standard loader

rag/
  classifier.py                  Alignment classifier
  llm_judge.py                   LLM judge
  prompts.py                     prompt templates

notebooks/                       analysis and visualisation
outputs/                         generated artifacts (indices, results, eval)
data/                            evidence HTML, gold standard, recommendations
```
