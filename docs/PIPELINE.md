# PIPELINE.md — RAG Pipeline for Greener Electronics Alignment

> This document is self-contained. It describes what the pipeline does, why each
> component exists, and how to run it end-to-end.
> For deeper methodological discussion and references, see [METHODOLOGY.md](METHODOLOGY.md).

---

## What This Pipeline Does

The pipeline takes **sustainability recommendations for the electronics sector**
(from academic papers or an internal whitepaper) and automatically answers the
question: *Is this recommendation already covered by EU legislation — and if so,
how?*

It does this in three stages:

1. **Retrieve** the most relevant EU legislative provisions for each recommendation.
2. **Classify** the degree of alignment (e.g. directly supported, only partially,
   conflicting, or not covered).
3. **Evaluate** the quality of both retrieval and classification against a
   hand-annotated gold standard.

All models are open-source and run locally — no external API calls.

---

## End-to-End Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUTS                                                         │
│  ├─ EU legislation HTML (EUR-Lex)    data/evidence/             │
│  └─ Recommendations (CSV / TXT)      data/recommendations*/     │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  STEP 1: CHUNKING   │  Parse HTML → one row per legal paragraph
        │  chunking_evidence  │  Parse TXT  → one atomic recommendation per row
        └──────────┬──────────┘
                   │  outputs/evidence.csv
        ┌──────────▼──────────┐
        │  STEP 2: INDEXING   │  Embed all chunks with BAAI/bge-m3 (1024-dim)
        │  embedding_indexing │  Build FAISS HNSW index  (dense search)
        │                     │  Build BM25Okapi index   (keyword search)
        └──────────┬──────────┘
                   │  outputs/indices/
        ┌──────────▼──────────────────────────────────────────┐
        │  STEP 3: RETRIEVAL  (run once per recommendation)   │
        │                                                      │
        │  Query = recommendation text                         │
        │                                                      │
        │  BM25Retriever  ──────────────────────┐             │
        │  (keyword match of query terms        │             │
        │   against provision text)             │ RRF fusion  │
        │                                       ├───────────► │
        │  DenseRetriever ──────────────────────┘  top-2k     │
        │  (cosine similarity between query                    │
        │   embedding and stored embeddings)                   │
        │                         │                            │
        │              ┌──────────▼──────────┐                │
        │              │  RerankedRetriever  │                │
        │              │  cross-encoder      │ ← optional     │
        │              │  scores each        │   2nd stage    │
        │              │  (query, chunk) pair│                │
        │              └──────────┬──────────┘                │
        │                         │  top-k final chunks        │
        └─────────────────────────┼────────────────────────────┘
                                  │
        ┌─────────────▼──────────────────────┐
        │  STEP 4: RAG CLASSIFICATION        │
        │  AlignmentClassifier               │
        │  (Qwen/Qwen2.5-7B-Instruct)        │
        │                                    │
        │  Input:  recommendation text       │
        │        + top-k retrieved chunks    │
        │  Output: alignment label           │
        │        + justification (free text) │
        │        + cited chunk IDs           │
        └──────────────┬─────────────────────┘
                       │
        ┌──────────────▼─────────────────────┐
        │  STEP 5: EVALUATION                │
        │                                    │
        │  Retrieval quality                 │
        │  ├─ Document-level  (gold standard)│
        │  └─ Paragraph-level (gold standard)│
        │                                    │
        │  Classification quality            │
        │  ├─ Accuracy, F1, Cohen's κ        │
        │  └─ LLM-as-Judge (Mistral-7B)      │
        └────────────────────────────────────┘
```

---

## Step 1 — Data Preparation

### EU Legislation Corpus

The evidence corpus consists of EU regulatory HTML documents downloaded from
EUR-Lex. `retrieval/chunking_evidence.py` parses the HTML DOM and extracts
every numbered paragraph within every article, preserving the full document
hierarchy (Document → Chapter → Article → Paragraph). Each chunk is output as
one row in `outputs/evidence.csv`.

**Why paragraph-level chunks?**  Splitting at the paragraph level keeps legal
clauses intact (sub-points a, b, c stay together) while keeping chunks short
enough to fit within the LLM context window and precise enough for accurate
retrieval scoring.

| Chunk Field | Description |
|-------------|-------------|
| `id` | Stable hash ID: `document\|article\|para\|hash` |
| `document` | Short name (e.g. `"ESPR"`, `"WEEE"`) |
| `source_file` | Original HTML filename |
| `chapter` | Chapter/title heading |
| `article` | Article number (e.g. `"Article 4"`) |
| `paragraph` | Paragraph number within the article |
| `text` | Full provision text |

```bash
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv
```

### Recommendations

Two recommendation sources are supported:

| Source | File | Format | Notes |
|--------|------|--------|-------|
| Gold standard (academic papers) | `data/gold_standard_doc_level/gold_standard.csv` | CSV | 273 entries, 13 papers, 19 EU docs — used for evaluation |
| Whitepaper | `data/recommendations_whitepaper/recommendations_empty.csv` | CSV (`;` delimiter) | 48 entries, 4 lifecycle phases — unlabelled, used for production |

The whitepaper CSV has columns `section`, `subsection`, `title`, `recommendation`.
The `recommendation` column is initially empty; the pipeline builds queries from
`section + subsection + title`.

---

## Step 2 — Index Building

`embedding_indexing.py` builds two search indices from the chunked evidence,
which are persisted to disk so that retrieval evaluation can be repeated cheaply.

### Embedding Model: `BAAI/bge-m3`

Each chunk text is embedded by `BAAI/bge-m3` (568 M parameters, 1024-dimensional
output). All embeddings are L2-normalised so that inner-product scores equal
cosine similarity. bge-m3 was selected for its top performance on the MTEB
Retrieval benchmark among models ≤1 B parameters, and its 8 192-token context
window which accommodates even the longest legislative paragraphs without
truncation.

### FAISS HNSW Index (dense search)

Stores all chunk embeddings in a Hierarchical Navigable Small World (HNSW) graph
for approximate nearest-neighbour search. HNSW provides sub-linear query time.

| Parameter | Value | Effect |
|-----------|-------|--------|
| `M` (bidirectional links) | 32 | Higher = better recall, more memory |
| `efConstruction` | 40 | Higher = better index quality, slower build |
| `efSearch` | 16 | Higher = better query recall, slower search |

### BM25 Index (sparse / keyword search)

Stores tokenised chunk texts as a `rank_bm25.BM25Okapi` object. Tokenisation
is whitespace splitting + lowercasing.

```bash
# Build indices from evidence CSV
python main.py build -i outputs/evidence.csv -m bge-m3
# Saved to: outputs/indices/bge-m3_faiss.index
#           outputs/indices/bge-m3_bm25.pkl
#           outputs/indices/bge-m3_chunks.pkl
```

---

## Step 3 — Retrieval

The goal of retrieval is to find the top-k EU legislative chunks most relevant
to a given recommendation text. Four strategies are implemented and compared.
All share the same interface: `retriever.retrieve(query, top_k) → RetrievalResult`.

### Strategy 1 — BM25 (Sparse / Keyword)

`retrieval/bm25_retriever.py` · class `BM25Retriever`

Scores chunks by how well the query's keyword terms appear in the provision text,
weighting by IDF and document length. Fast and interpretable, but cannot handle
paraphrases or synonyms (e.g. "extended producer responsibility" will not match
"EPR" unless both forms appear).

### Strategy 2 — Dense FAISS (Semantic)

`retrieval/dense_retriever.py` · class `DenseRetriever`

Encodes the query with bge-m3, then retrieves the nearest chunk embeddings from
the FAISS index. Captures semantic similarity and handles paraphrases well, but
can miss exact regulatory terminology that BM25 handles trivially.

### Strategy 3 — Hybrid RRF (Combined)

`retrieval/hybrid_retriever.py` · class `HybridRetriever`

Runs BM25 and Dense retrieval independently (each retrieving `2 × top_k`
candidates), then merges the two ranked lists using **Reciprocal Rank Fusion**:

```
RRF_score(chunk) = Σ  1 / (60 + rank_in_list)
                 lists
```

The constant 60 prevents a single top-ranked item from dominating. RRF is
parameter-free and consistently outperforms individual retrieval strategies
across diverse retrieval tasks. This is the recommended default strategy.

### Strategy 4 — Reranker (Two-Stage)

`retrieval/reranker.py` · classes `Reranker`, `RerankedRetriever`

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22 M parameters)
re-scores the top-2k candidates from any base retriever. Unlike bi-encoders
(which encode query and document separately), the cross-encoder processes the
`(query, chunk)` pair jointly with full bidirectional attention, enabling much
finer relevance judgements. This improves precision at low k values
significantly, at the cost of higher latency.

Any of the three base strategies can be wrapped: BM25+Reranker,
Dense+Reranker, or Hybrid+Reranker (recommended).

### Retrieval Comparison: 6 Configurations

| # | Configuration | Base | Reranker |
|---|---------------|------|----------|
| 1 | BM25 | ✓ | — |
| 2 | Dense | ✓ | — |
| 3 | Hybrid (RRF) | ✓ | — |
| 4 | BM25 + Reranker | ✓ | ✓ |
| 5 | Dense + Reranker | ✓ | ✓ |
| 6 | Hybrid + Reranker | ✓ | ✓ ← default |

All six are evaluated in `notebooks/01_retrieval_analysis.ipynb`.

---

## Step 4 — RAG Alignment Classification

`rag/classifier.py` · class `AlignmentClassifier`

Once relevant chunks are retrieved, the classifier decides *how* the recommendation
relates to those provisions.

**Model:** `Qwen/Qwen2.5-7B-Instruct` (7.6 B parameters, Apache 2.0 licence).
Selected for reliable structured JSON output, strong instruction-following, and
efficient inference at 7 B scale.

| Property | Value |
|----------|-------|
| Context length | 32 768 tokens |
| Temperature | 0.0 (deterministic) |
| VRAM (fp16) | ~15 GB |
| VRAM (4-bit quantised) | ~5 GB |

### How Classification Works

1. The recommendation text and the top-k retrieved chunks are assembled into a
   prompt (see `rag/prompts.py`).
2. The model is instructed to analyse the evidence and output strict JSON:

```json
{
  "label": "<alignment label>",
  "justification": "<evidence-based reasoning>",
  "cited_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}
```

3. The response is parsed; label is fuzzy-matched to the canonical label list
   to tolerate minor output variations.

### Alignment Labels

| Label | Meaning |
|-------|---------|
| **Aligned** | The recommendation is directly supported by existing EU law. |
| **Conditional** | Alignment depends on delegated or implementing acts not yet in force. |
| **Conflicting** | The recommendation contradicts current legislative provisions. |
| **No explicit legal basis** | No EU provision addresses the recommendation at all. |
| **Beyond compliance** | The recommendation goes further than what current legislation requires. |

---

## Step 5 — Evaluation

The pipeline uses a three-stage evaluation framework.

### Stage 1 — Document-Level Retrieval (Gold Standard)

*Can the retriever find the correct EU legislation document for a given recommendation?*

The gold standard (`data/gold_standard_doc_level/gold_standard.csv`) contains
273 manually annotated entries from 13 academic papers, each linking a
recommendation to one or more of 19 EU regulatory documents.

At this level, retrieved chunks are deduplicated to their parent document names
before scoring: only unique documents count. Metrics are reported at
k ∈ {1, 3, 5, 10, 20}.

### Stage 2 — Paragraph-Level Retrieval (Gold Standard)

*Does the retriever surface the right provisions, or does it waste top-k positions
on irrelevant chunks from irrelevant documents?*

Each chunk is scored individually. A chunk is **relevant** if its parent document
appears in the gold-standard set for that query. This is strictly harder than
document-level: a retriever that floods the top-k with chunks from the wrong
legislation will be penalised, even if the correct law appears at rank k+1.

### Stage 3 — Whitepaper Evaluation (Unlabelled / Human Evaluation)

*Does the pipeline produce sensible results on the 48 real whitepaper recommendations?*

Since no gold labels exist for the whitepaper, outputs are exported to a CSV
structured for human review (`human_label` and `human_notes` columns are left
empty). Two evaluators assign independent labels to enable inter-annotator
agreement (Cohen's κ or Krippendorff's α).

### Retrieval Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Hit@k** | Binary — did at least one relevant item appear in top-k? |
| **Recall@k** | Coverage — what fraction of all relevant items were found? |
| **Precision@k** | Signal/noise — what fraction of the top-k is actually relevant? |
| **MRR** | Speed — reciprocal rank of the first relevant result (higher = found faster) |
| **MAP** | Average Precision over all relevant items (position-sensitive) |
| **NDCG@k** | Ranking quality with logarithmic discount for depth |

### Statistical Robustness

To ensure results meet publication standards:

- **95% Bootstrap confidence intervals** (10 000 resamples, percentile method):
  per-query score vectors are resampled with replacement; the 2.5th–97.5th
  percentile of the distribution defines the interval.
- **Paired permutation tests** (10 000 permutations, two-sided): assesses
  whether the difference between two retrieval configurations is statistically
  significant. Differences with p < 0.05 are flagged with ★.

### Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correct label predictions |
| Macro-F1 | Unweighted average F1 across all labels (penalises rare label failures equally) |
| Weighted-F1 | Support-weighted F1 (reflects class frequency) |
| Cohen's κ | Agreement beyond chance, robust to class imbalance |
| Per-class P/R/F1 | Precision, recall, F1 for each individual alignment label |
| Confusion matrix | True vs. predicted label counts |

### LLM-as-Judge

`rag/llm_judge.py` · class `LLMJudge`

A second LLM (`mistralai/Mistral-7B-Instruct-v0.3`, different architecture from
the classifier) independently evaluates each classification on three dimensions:

| Dimension | Scale | What It Checks |
|-----------|-------|----------------|
| Label correctness | 1–5 | Is the assigned alignment label appropriate given the evidence? |
| Justification quality | 1–5 | Is the reasoning thorough, specific, and grounded in cited provisions? |
| Evidence usage | 1–5 | Are the cited chunks actually relevant, or was irrelevant text cited? |

The **overall score** is the mean of the three sub-scores. Using a different
model family mitigates self-evaluation bias (models tend to rate their own output
more favourably).

---

## Configuration Reference (`config.py`)

All paths, model names, and hyperparameters are centralised in `config.py`.
Key settings:

| Constant | Default Value | Description |
|----------|---------------|-------------|
| `DEFAULT_MODEL_KEY` | `"bge-m3"` → `BAAI/bge-m3` | Embedding model (568 M params, 1024-dim) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Alignment classifier |
| `JUDGE_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | LLM-as-judge |
| `DEFAULT_TOP_K` | 10 | Final number of chunks passed to the classifier |
| `DEFAULT_RERANK_TOP` | 5 | Top-k after reranking (≤ DEFAULT_TOP_K) |
| `RRF_K` | 60 | RRF smoothing constant |
| `EVAL_K_VALUES` | `[1, 3, 5, 10, 20]` | Cut-off depths for retrieval metrics |
| `FAISS_HNSW_M` | 32 | HNSW bidirectional links per node |
| `FAISS_EF_CONSTRUCT` | 40 | HNSW build-time quality parameter |
| `FAISS_EF_SEARCH` | 16 | HNSW query-time recall parameter |

---

## Data Models (`data_models.py`)

Shared dataclasses passed between all pipeline stages:

| Class | Represents | Key Fields |
|-------|-----------|------------|
| `Chunk` | One legal paragraph | `id`, `document`, `article`, `paragraph`, `text` |
| `Recommendation` | One sustainability recommendation | `section`, `subsection`, `title`, `text` |
| `GoldStandardEntry` | One annotated (recommendation → EU doc) pair | `recommendation_text`, `relevant_document`, `alignment_label` |
| `RetrievalResult` | Output of any retriever | `query`, `ranked_chunks: list[Chunk]`, `scores: list[float]` |
| `ClassificationResult` | Output of the classifier | `label`, `justification`, `cited_chunk_ids`, `retrieved_chunks` |
| `RetrievalMetrics` | Aggregated retrieval scores at one k | `hit_rate`, `recall`, `precision`, `mrr`, `map_score`, `ndcg` |
| `ClassificationMetrics` | Aggregated classification scores | `accuracy`, `macro_f1`, `weighted_f1`, `cohens_kappa` |

---

## Running the Pipeline

### Environment Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows
# source venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB (4-bit quantised) | 16+ GB (fp16) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB (model weights) | 40 GB |

### CLI Commands

```bash
# 1. Chunk evidence HTML into CSV
python retrieval/chunking_evidence.py -i data/evidence -o outputs/evidence.csv

# 2. Build FAISS + BM25 indices
python main.py build -i outputs/evidence.csv -m bge-m3

# 3. Evaluate retrieval against the gold standard
python main.py evaluate --gold data/gold_standard_doc_level/gold_standard.csv

# 4. Classify recommendations
python main.py classify -i outputs/recommendations.csv -o outputs/classifications.csv

# 5. Full pipeline: retrieve → classify → evaluate
python main.py run -i outputs/recommendations.csv \
                   --gold data/gold_standard_doc_level/gold_standard.csv
```

### Notebooks (recommended for research)

Run in order from the `notebooks/` directory:

| # | Notebook | What It Does |
|---|----------|-------------|
| 1 | `01_retrieval_analysis.ipynb` | Compares all 6 retrieval configurations at document + paragraph level; bootstrap CIs; paired significance tests; error analysis |
| 2 | `02_rag_evaluation.ipynb` | Runs alignment classification; LLM-as-judge scoring; score distributions by label |
| 3 | `03_evaluation_metrics.ipynb` | Consolidated breakdown of all evaluation metrics |
| 4 | `04_whitepaper_evaluation.ipynb` | Applies the pipeline to the 48 whitepaper recommendations; exports a CSV ready for human annotation |

---

## File Map

```
├── config.py                  # All paths, models, and hyperparameters
├── data_models.py             # Shared dataclasses: Chunk, Recommendation, metrics
├── embedding_indexing.py      # Embed chunks, build FAISS + BM25 indices
├── pipeline.py                # CLI sub-commands: build / evaluate / classify / run
├── main.py                    # Entry point (delegates to pipeline.py)
│
├── retrieval/
│   ├── base_retriever.py           # BaseRetriever ABC (retrieve interface)
│   ├── bm25_retriever.py           # BM25Retriever — keyword search
│   ├── dense_retriever.py          # DenseRetriever — FAISS semantic search
│   ├── hybrid_retriever.py         # HybridRetriever — BM25 + FAISS + RRF fusion
│   ├── reranker.py                 # Reranker (cross-encoder) + RerankedRetriever wrapper
│   ├── chunking_evidence.py        # EUR-Lex HTML → paragraph-level chunk CSV
│   ├── chunking_recommendations.py # Structured TXT → atomic recommendation CSV
│   └── retrieval.py                # Low-level search utilities
│
├── rag/
│   ├── classifier.py          # AlignmentClassifier (Qwen2.5-7B): retrieve → label
│   ├── llm_judge.py           # LLMJudge (Mistral-7B): score label/justification/evidence
│   └── prompts.py             # All prompt templates + message builders
│
├── evaluation/
│   ├── evaluation.py          # Load gold standard; doc-level + para-level evaluation
│   └── metrics.py             # Hit@k, Recall, NDCG, MAP, F1, bootstrap CI, permutation test
│
├── notebooks/
│   ├── 01_retrieval_analysis.ipynb      # Retrieval comparison + statistical analysis
│   ├── 02_rag_evaluation.ipynb          # Classification + LLM judge
│   ├── 03_evaluation_metrics.ipynb      # Consolidated metric breakdown
│   └── 04_whitepaper_evaluation.ipynb   # Whitepaper → human-evaluation export
│
├── docs/
│   ├── PIPELINE.md            # This file — pipeline walkthrough and reference
│   └── METHODOLOGY.md         # Detailed prose methodology with citations
│
├── data/
│   ├── evidence/                              # EUR-Lex HTML documents
│   ├── gold_standard_doc_level/
│   │   └── gold_standard.csv                 # 273 annotated recommendation→doc pairs
│   └── recommendations_whitepaper/
│       └── recommendations_empty.csv         # 48 whitepaper recommendations (unlabelled)
│
└── outputs/
    ├── evidence.csv                           # Chunked evidence (one row per paragraph)
    └── indices/
        ├── bge-m3_faiss.index                 # FAISS HNSW index
        ├── bge-m3_bm25.pkl                    # BM25Okapi index
        └── bge-m3_chunks.pkl                  # Chunk list (aligned with index positions)
```
