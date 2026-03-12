# Benchmarks

This folder contains benchmark datasets for evaluating retrieval quality
on external (out-of-domain) data, complementing the in-domain gold-standard
evaluation.

## Expected Format

Each benchmark is a JSON file with the schema:

```json
[
  {
    "query": "natural-language query text",
    "relevant_ids": ["chunk_id_1", "chunk_id_2"],
    "relevant_docs": ["document_name_1"]
  }
]
```

- **`relevant_ids`** — chunk-level relevance (for paragraph-level evaluation).
- **`relevant_docs`** — document-level relevance (for document-level evaluation).
  Use the `id_field` parameter in `evaluate_benchmark()` to choose.

## Generating the In-Domain Benchmark

Run the provided script to generate a benchmark from the gold standard:

```bash
python benchmarks/generate_benchmark.py
```

This creates `benchmarks/gold_standard_benchmark.json` — a held-out test
split that can be used for benchmarking without data leakage concerns.

## Recommended English-Language Benchmarks for Contextualisation

The following MTEB / BEIR benchmarks are relevant for contextualising
retrieval performance on legal and regulatory text in English:

| Dataset | Source | Domain | Why Relevant |
|---------|--------|--------|--------------|
| **LegalBench** – ConsumerContractsQA | [MTEB](https://huggingface.co/datasets/mteb/LegalBenchConsumerContractsQA) | English legal | Closest text register to EU legislation |
| **LegalBench** – CorporateLobbying | [MTEB](https://huggingface.co/datasets/mteb/LegalBenchCorporateLobbying) | English legal | Regulatory-adjacent topic |
| **MSMARCO** | [BEIR](https://huggingface.co/datasets/BeIR/msmarco) | English web/general | Large-scale general-purpose retrieval |
| **NFCorpus** | [BEIR](https://huggingface.co/datasets/BeIR/nfcorpus) | English biomedical | Cross-domain professional text |
| **SciFact** | [BEIR](https://huggingface.co/datasets/BeIR/scifact) | English scientific | Claim verification against evidence |
| **FiQA** | [BEIR](https://huggingface.co/datasets/BeIR/fiqa) | English financial | Regulatory-adjacent financial QA |

### How to Use MTEB Scores for Contextualisation

Rather than running these full benchmarks locally, compare your pipeline's
NDCG@10 against the published MTEB leaderboard scores for the same
embedding model. For example:

| Model | MTEB Retrieval Avg | LegalBenchConsumerContractsQA | Our Gold Std NDCG@10 |
|-------|-------------------|------------------------------|---------------------|
| BAAI/bge-m3 | ~0.599 | — | (your score) |
| all-mpnet-base-v2 | ~0.501 | — | (your score) |

This contextualisation establishes whether your pipeline's performance is
within the expected range for the embedding model on domain-specific text.

## Running a Benchmark

```bash
python pipeline.py benchmark -i benchmarks/gold_standard_benchmark.json \
                             -o outputs/benchmark_metrics.json
```
