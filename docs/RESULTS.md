# Results Summary

This document summarises the results produced by the unified evaluation workflow in `outputs/eval_unified/`. It focuses on the latest consolidated run executed with three embedding models (`minilm`, `mpnet`, `bge-m3`), top-k retrieval evaluation at `k ∈ {1, 3, 5, 10, 20}`, reranking enabled, and MTEB evaluation limited to a `--max-corpus 3000` subset for a tractable comparison run.

## 1. What Was Evaluated

The current workflow evaluates the retrieval stage of the thesis pipeline in three ways:

1. **Gold-standard document-level retrieval**
   The system retrieves EU evidence chunks for each annotated recommendation and is scored after deduplicating chunks to their parent documents.

2. **MTEB LegalBench chunk-level retrieval**
   The system is evaluated on an external legal retrieval benchmark at chunk level.

3. **Whitepaper retrieval export**
   Retrieved evidence is exported for the whitepaper recommendations. This stage is not scored automatically in the current unified run because no ground-truth labels are provided.

The relevant output files are:

- `outputs/eval_unified/metrics_all.csv`
- `outputs/eval_unified/ranking_k10.csv`
- `outputs/eval_unified/run_summary.json`
- `outputs/eval_unified/*retrieved_chunks*.csv`

## 2. Evaluation Setup

### Models

- `all-MiniLM-L6-v2` (`minilm`)
- `all-mpnet-base-v2` (`mpnet`)
- `BAAI/bge-m3` (`bge-m3`)

### Retrieval Methods

- `bm25`: lexical sparse retrieval
- `dense`: semantic embedding retrieval
- `rrf`: hybrid fusion of BM25 and dense retrieval via Reciprocal Rank Fusion
- `*_rerank`: the same first-stage retrievers followed by a cross-encoder reranker

### Metrics

For the gold-standard run, the main metrics are:

- `Hit@k`: whether at least one correct result is found in the top-k
- `MRR`: how early the first relevant result appears
- `NDCG@k`: ranking quality with stronger credit for earlier correct hits
- `Mean Rank`: average position of the first relevant item, where lower is better

The latest unified run scored:

- `227` gold-standard queries
- `396` MTEB LegalBench queries

## 3. Main Results

### 3.1 Gold Standard: Best Document-Level Results at k = 10

The strongest gold-standard results at `k=10` are shown below.

| Goal | Best configuration | Hit@10 | MRR | NDCG@10 | Mean Rank |
|---|---|---:|---:|---:|---:|
| Best recall / coverage | `bge-m3 + rrf` | 0.7093 | 0.5954 | 0.6236 | 1.4161 |
| Best ranking quality | `bm25_rerank` | 0.6960 | 0.6002 | 0.6247 | 1.3734 |

### Interpretation

- The best **coverage-oriented** setup in the current run is the hybrid first-stage retriever with `bge-m3`, which reaches `Hit@10 = 0.7093`. This means the correct document appears in the top 10 for about 71% of gold-standard queries.
- The best **ranking-oriented** setup is `bm25_rerank`, which achieves the highest `MRR = 0.6002`, the highest `NDCG@10 = 0.6247`, and the lowest `Mean Rank = 1.3734`.
- This pattern suggests that lexical matching remains very strong in this domain, and reranking helps place the relevant legal document earlier in the ranked list.
- At the same time, hybrid retrieval slightly improves the chance of recovering the correct document somewhere in the top-k, which is useful when downstream review can inspect multiple retrieved candidates.

### 3.2 Gold Standard: Model Comparison

Looking only at the dense and hybrid methods, the model differences are real but modest.

At `k=10`:

- `bge-m3 + dense`: `MRR = 0.5969`, `NDCG@10 = 0.6205`
- `mpnet + dense`: `MRR = 0.5716`, `NDCG@10 = 0.6071`
- `minilm + dense`: `MRR = 0.5628`, `NDCG@10 = 0.5987`

This indicates that `bge-m3` is the strongest pure dense retriever on the in-domain benchmark.

For hybrid retrieval without reranking:

- `bge-m3 + rrf`: `Hit@10 = 0.7093`, `MRR = 0.5954`, `NDCG@10 = 0.6236`
- `mpnet + rrf`: `Hit@10 = 0.7004`, `MRR = 0.5888`, `NDCG@10 = 0.6167`
- `minilm + rrf`: `Hit@10 = 0.7093`, `MRR = 0.5896`, `NDCG@10 = 0.6203`

The in-domain conclusion is therefore:

- `bge-m3` is the best dense model overall.
- Hybrid retrieval is slightly more robust than dense-only retrieval.
- Cross-encoder reranking improves the position of relevant documents, but it does not uniformly increase Hit@10 in this run.

### 3.3 MTEB LegalBench: Chunk-Level Results at k = 10

For the external benchmark, the unified run reports `rrf_rerank` results for each embedding model.

| Model | Hit@10 | MRR | NDCG@10 | Mean Rank |
|---|---:|---:|---:|---:|
| `mpnet + rrf_rerank` | 0.9242 | 0.6814 | 0.7405 | 2.0956 |
| `minilm + rrf_rerank` | 0.9167 | 0.6781 | 0.7363 | 2.0579 |
| `bge-m3 + rrf_rerank` | 0.9015 | 0.6758 | 0.7308 | 2.0196 |

### Interpretation

- `mpnet` performs best on the evaluated MTEB subset for both `Hit@10` and `NDCG@10`.
- The differences between the three models are relatively small, which suggests that all three are competitive on this external legal benchmark.
- The ordering here differs slightly from the in-domain gold-standard results, where `bge-m3` is stronger. That is useful evidence that model choice depends on the target distribution rather than on public benchmark averages alone.

## 4. Whitepaper Outputs

The current unified run exports retrieved evidence for the whitepaper recommendations but does not assign automatic retrieval scores because the whitepaper set is unlabelled.

Generated files include:

- `outputs/eval_unified/whitepaper_retrieved_chunks_bge-m3_rrf_rerank.csv`
- `outputs/eval_unified/whitepaper_retrieved_chunks_mpnet_rrf_rerank.csv`
- `outputs/eval_unified/whitepaper_retrieved_chunks_minilm_rrf_rerank.csv`

These files are the basis for qualitative analysis. They let you inspect:

- which provisions were retrieved for each recommendation
- whether the evidence looks legally relevant
- whether different retrievers surface different policy rationales

For the thesis, this stage should be framed as an application and inspection stage rather than a benchmarked score table.

## 5. Overall Evaluation of the Obtained Results

The current results support four main conclusions.

### 1. The retrieval pipeline is effective but not saturated

On the gold-standard benchmark, the best configuration retrieves the correct EU document in the top 10 for about 70% of queries. This is strong enough to justify the retrieval-first RAG design, but it also shows there is still room to improve recall.

### 2. Domain-specific lexical matching remains important

The best rank-sensitive gold-standard result comes from `bm25_rerank`, not from dense-only retrieval. In this corpus, legal terminology and exact wording still carry substantial signal.

### 3. Hybrid retrieval is the safest default first stage

Hybrid retrieval consistently performs near the top and gives the best or near-best coverage. That makes it a strong default retrieval strategy when the goal is to support downstream interpretation or human review.

### 4. Public benchmark winners are not automatically in-domain winners

On the MTEB subset, `mpnet` is slightly best in this run, whereas `bge-m3` is strongest on the in-domain gold-standard benchmark. The thesis should therefore treat MTEB as contextual evidence, not as the sole basis for model selection.

## 6. Recommended Thesis Framing

You can present the results with the following argument:

- The pipeline is evaluated on both an in-domain benchmark and an external legal benchmark.
- The in-domain benchmark is the primary basis for model and method selection.
- `bge-m3 + rrf` is the best first-stage retriever if the priority is finding the right document within the inspected set.
- `bm25_rerank` is the best configuration if the priority is pushing the relevant document to the very top of the ranking.
- Whitepaper results demonstrate applicability and provide case-study evidence, but they are not a scored benchmark in the current run.

## 7. Limitations of the Current Result Set

- The MTEB evaluation was run with `--max-corpus 3000`, so these are controlled comparison results rather than a final full-corpus benchmark report.
- The unified run reports retrieval results only. If classification results are to be discussed in the thesis results chapter, they should be generated and summarised separately.
- The gold-standard metrics in the current consolidated outputs are based on `227` evaluated queries, so thesis text should cite the executed run outputs directly when presenting these tables.

## 8. Files to Cite in the Thesis

For reporting and figure generation, the main result sources are:

- `outputs/eval_unified/metrics_all.csv`
- `outputs/eval_unified/ranking_k10.csv`
- `notebooks/05_eval_visualizations_only.ipynb`

These files contain the numerical tables, model rankings, and plots used to support the conclusions above.

