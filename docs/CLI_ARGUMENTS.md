# CLI Arguments Reference

This file lists all currently supported CLI arguments from `main.py` / `pipeline.py`.

## Command: build

Usage:

```powershell
python main.py build [options]
```

Arguments:

- `-i`, `--input` (required, one or more paths)
  - Evidence CSV file(s) to index.
- `-m`, `--model` (optional)
  - Embedding model key.
  - Default: `bge-m3`
  - Choices are from `EMBEDDING_MODELS` in `config.py`.

## Command: prompt

Usage:

```powershell
python main.py prompt [options]
```

Arguments:

- `-i`, `--input` (optional)
  - Recommendations CSV input.
  - Default: `data/recommendations_whitepaper/recommendations_v2.csv`
- `-o`, `--output` (optional)
  - Output CSV path for prompt/classification rows.
  - Default: `<OUTPUT_DIR>/prompt_results.csv`
- `-m`, `--model` (optional)
  - Embedding model key for retrieval index.
  - Default: `bge-m3`
- `-k`, `--top-k` (optional)
  - Candidate retrieval size before reranking.
  - Default: `10`
- `--rerank-top` (optional)
  - Final result size after reranking.
  - Default: `5`
- `--retrieval-mode` (optional)
  - Evidence retrieval mode.
  - Default: `flat_baseline`
  - Choices:
    - `flat_baseline`
    - `split_evidence_retrieval`
- `--max-chunks-per-doc` (optional)
  - Anti-dominance cap on chunks per document in final set (split mode).
  - Default: `2`
- `--near-dup-suppression` (flag)
  - Enable coarse same-document near-duplicate suppression (split mode).
- `--no-rerank` (flag)
  - Disable cross-encoder reranker.
- `--retrieve-only` (flag)
  - Skip classification stage (retrieval export only).
- `--judge` (flag)
  - Run LLM-as-judge on classifier outputs.

Notes:

- The prompt retriever is hybrid fusion internally (BM25 + dense + RRF).
- `fusion` is not a `--model` value. Fusion is the retrieval method, not the embedding key.

## Command: evaluate

Usage:

```powershell
python main.py evaluate [options]
```

Arguments:

- `--models` (optional, one or more model keys)
  - Models to compare.
  - Default: `bge-m3 e5-large-v2 e5-mistral`
- `--include-splade` (flag)
  - Include SPLADE sparse baseline.
- `--splade-model` (optional)
  - SPLADE model id.
  - Default: `naver/splade-cocondenser-ensembledistil`
- `--splade-max-length` (optional)
  - Max token length for SPLADE.
  - Default: `256`
- `--gold-csv` (optional)
  - Gold standard CSV path.
- `--whitepaper-csv` (optional)
  - Whitepaper recommendation CSV path.
- `--mteb-dataset` (optional)
  - MTEB dataset id.
  - Default: `mteb/legalbench_consumer_contracts_qa`
- `--mteb-split` (optional)
  - MTEB split name.
  - Default: `test`
- `--max-corpus` (optional int)
  - Limit MTEB corpus size.
- `--full-mteb` (flag)
  - Run on full MTEB corpus.
- `--top-k` (optional int)
  - Retrieval top-k.
  - Default: `10`
- `--rerank-top` (optional int)
  - Reranked top-k.
  - Default: `5`
- `--export-k` (optional int)
  - Export cut-off k.
  - Default: `10`
- `--k-values` (optional list of ints)
  - Evaluation k values.
  - Default: `1 3 5 10 20`
- `--output-dir` (optional path)
  - Unified evaluation output directory.
- `--skip-reranker` (flag)
  - Disable reranker in evaluation.
- `--force-cpu` (flag)
  - Force CPU execution.
- `--skip-mteb` (flag)
  - Skip MTEB phase.
- `--skip-whitepaper` (flag)
  - Skip whitepaper phase.
- `--auto-build-indices` (flag)
  - Build missing indices automatically.
- `--evidence-csv` (optional path)
  - Evidence CSV for auto-index build.
- `--with-robustness` (flag)
  - Run robustness analysis.
- `--robust-model` (optional)
  - Model key for robustness run.
- `--robust-k` (optional int)
  - Robustness k.
  - Default: `10`

## Command: download-models

Usage:

```powershell
python main.py download-models [options]
```

Arguments:

- `--embedding-models` (optional, one or more keys)
  - Models to pre-download.
  - Default: `bge-m3 e5-large-v2 e5-mistral`
- `--include-llms` (flag)
  - Also pre-download classifier/judge LLMs.
