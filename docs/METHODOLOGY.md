# Pipeline Documentation — RAG for Aligning Greener Electronics Recommendations

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement & Research Design](#2-problem-statement--research-design)
3. [Data Pipeline](#3-data-pipeline)
4. [Embedding Model Selection](#4-embedding-model-selection)
5. [Retrieval Module](#5-retrieval-module)
6. [RAG Alignment Classification](#6-rag-alignment-classification)
7. [LLM-as-Judge Evaluation](#7-llm-as-judge-evaluation)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Publishability Considerations](#9-publishability-considerations)
10. [Reproducibility](#10-reproducibility)
11. [File Reference](#11-file-reference)
12. [References](#12-references)

---

## 1. Introduction

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that
systematically evaluates how sustainability recommendations for the electronics
sector align with the European Union's regulatory framework.  The pipeline takes
free-text recommendations — drawn from academic literature and a dedicated
whitepaper — and (i) retrieves the most relevant EU legislative provisions,
(ii) classifies the degree of alignment using an open-source large language model,
and (iii) evaluates the quality of both retrieval and classification through a
multi-tier evaluation framework.

The motivation is twofold.  First, the volume of EU regulatory text relevant to
green electronics (ESPR, REACH, RoHS, WEEE, Battery Regulation, and many more)
makes manual cross-referencing impractical.  A reliable automated pipeline can
accelerate this mapping for researchers and policymakers alike.  Second, by using
only open-source, locally hosted models, the pipeline ensures full reproducibility,
eliminates per-token API costs, and keeps potentially sensitive recommendation
texts off external servers.

---

## 2. Problem Statement & Research Design

### 2.1 Research Question

*To what extent can a hybrid retrieval-augmented generation pipeline
automatically identify and classify the alignment of sustainability
recommendations with the EU regulatory framework for electronics?*

### 2.2 Three-Stage Evaluation Methodology

The evaluation is organised in three stages, each providing a different lens on
pipeline quality:

**Stage 1 — Document-level retrieval (gold standard).**
The first question is: given a sustainability recommendation, can the retriever
identify the correct EU legislation document?  We answer this using a
hand-annotated gold standard (`gold_standard.csv`) containing 273 entries across
13 academic papers, each linking a recommendation or statement to one of 19 EU
regulatory documents.  Retrieval quality is measured at multiple cut-off depths
(k ∈ {1, 3, 5, 10, 20}) using standard information-retrieval metrics: Hit@k,
Recall@k, Precision@k, MRR, MAP, and NDCG@k.  For document-level evaluation,
retrieved chunks are deduplicated to their parent documents before scoring.

**Stage 2 — Paragraph-level retrieval (gold standard).**
Document-level evaluation tells us whether the right law was found, but not
whether the right *provisions* were surfaced.  Paragraph-level (chunk-level)
evaluation treats each retrieved chunk individually.  A chunk is defined as
relevant if its parent document appears in the gold-standard set for that
query.  This is a stricter test: it penalises retrievers that waste top-k
positions on chunks from irrelevant legislation, even if the correct document
appears elsewhere in the ranked list.  The same six metrics are reported at the
same cut-off depths.

**Stage 3 — Whitepaper recommendation evaluation (unlabelled).**
The whitepaper contains 48 sustainability recommendations spanning four lifecycle
phases (Design & Material Selection, Manufacturing, Use, End-of-Life).  These
have no ground-truth labels.  The pipeline retrieves evidence and classifies
alignment for each recommendation, and the outputs are exported to a CSV
structured for human evaluation.  This stage demonstrates the pipeline's
practical applicability and produces artefacts for qualitative assessment,
including inter-annotator agreement analysis if multiple evaluators are used.

### 2.3 Experimental Design (Retrieval Comparison)

To determine the best retrieval configuration, six retrieval strategies are
compared factorially:

| # | Base Retriever | Reranker | Description |
|---|---|---|---|
| 1 | BM25 | — | Sparse lexical baseline |
| 2 | Dense (FAISS) | — | Semantic similarity via bge-m3 |
| 3 | Hybrid (RRF) | — | BM25 + FAISS fused with Reciprocal Rank Fusion |
| 4 | BM25 | Cross-encoder | BM25 + ms-marco-MiniLM reranking |
| 5 | Dense (FAISS) | Cross-encoder | FAISS + ms-marco-MiniLM reranking |
| 6 | Hybrid (RRF) | Cross-encoder | Full pipeline (recommended) |

Each configuration is evaluated on the same query set at both document and
paragraph level.  Metrics are accompanied by 95 % bootstrap confidence intervals
(10 000 resamples, percentile method) and pairwise statistical significance is
assessed via two-sided paired permutation tests (10 000 permutations).

---

## 3. Data Pipeline

### 3.1 Evidence Corpus (EU Legislation)

The evidence corpus consists of consolidated texts from six core EU regulatory
documents obtained as HTML from EUR-Lex, supplemented by 19 additional
recommendation and strategy documents.  Together these cover the principal
legislation relevant to greener electronics: the Ecodesign for Sustainable
Products Regulation (ESPR), REACH, RoHS, WEEE Directive, Battery Regulation,
Waste Framework Directive, and associated strategies.

**Chunking** (`retrieval/chunking_evidence.py`) parses the HTML DOM using
BeautifulSoup, extracting the legal hierarchy — Document → Chapter/Title →
Article → Paragraph.  Each chunk corresponds to one numbered paragraph within
an article, preserving the full hierarchical context.  List items remain
attached to their parent paragraph to preserve the integrity of legal clauses
(e.g. article sub-points (a), (b), (c) stay together).  The output is a flat
CSV (`outputs/evidence.csv`) with one row per provision.

Each chunk carries the following fields:

| Field | Description |
|-------|-------------|
| `id` | Stable hash-based identifier (`document\|article\|para\|hash`) |
| `document` | Short document name (e.g. "ESPR", "WEEE") |
| `source_file` | Original HTML filename |
| `chapter` | Chapter/title heading |
| `article` | Article number (e.g. "Article 4") |
| `paragraph` | Paragraph number within the article |
| `text` | Full provision text |

### 3.2 Recommendations

Two recommendation sources are used:

1. **Gold-standard recommendations** — 273 entries extracted from 13
   academic papers, each annotated with the relevant EU document.  These
   serve as the evaluation benchmark.

2. **Whitepaper recommendations** — 48 entries across four lifecycle
   phases, with columns `section`, `subsection`, `title`, and
   `recommendation` (the last currently empty; the title and
   section/subsection context form the retrieval query).

The whitepaper CSV uses semicolons (`;`) as delimiters.

### 3.3 Index Building

**Embedding** (`embedding_indexing.py`) encodes all chunk texts using a
sentence-transformer model (default: BAAI/bge-m3, 568 M parameters, 1024
dimensions).  Embeddings are L2-normalised so that inner product scores
correspond to cosine similarity.

**Token-limit awareness.**  Before encoding, the pipeline audits all chunk
texts against the model's maximum token limit (8 192 tokens for bge-m3,
512 for mpnet / MiniLM).  Chunks that exceed the limit — whose tail tokens
would be silently truncated by the tokenizer — are flagged with a console
warning reporting the count and percentage of over-length texts.  This
transparency allows the user to adjust chunking parameters if needed.
In practice, paragraph-level EU legislative chunks rarely exceed even 512
tokens, so truncation is unlikely with the default settings.

**FAISS index** — a Hierarchical Navigable Small World (HNSW) graph is built
for approximate nearest-neighbour search.  HNSW provides sub-linear query time
while maintaining high recall.  The parameters are tuned for a small corpus:
M = 32 bidirectional links per node, efConstruction = 40, efSearch = 16.

**BM25 index** — a BM25Okapi sparse lexical index is built over
whitespace-tokenised, lowercased chunk texts and stored as a pickled
`rank_bm25.BM25Okapi` object.

Both indices are persisted to disk (`outputs/indices/`) and loaded at
evaluation time, ensuring that index building (expensive) is decoupled from
retrieval evaluation (cheap).

---

## 4. Embedding Model Selection

### 4.1 The MTEB Benchmark

The Massive Text Embedding Benchmark (MTEB; Muennighoff et al., 2023) provides
the most comprehensive public evaluation framework for text embedding models.
It spans 8 task types across 113+ datasets in 56+ languages.  For this project,
the relevant task type is **Retrieval**: given a natural-language query, rank a
set of passages by relevance.

Among MTEB's retrieval datasets, three are particularly informative for
contextualising our results in a legal/regulatory setting:

| MTEB Dataset | Task | Why Relevant |
|---|---|---|
| LegalBenchConsumerContractsQA | legal clause retrieval | closest text register to EU legislation |
| LegalBenchCorporateLobbying | regulatory document retrieval | regulatory-adjacent legal topic |
| NFCorpus | biomedical passage retrieval | cross-domain professional text |

All three datasets are in English, matching the language of both our
recommendation queries and the EU legislative provisions in our evidence
corpus.  We report our pipeline's NDCG@10 alongside published MTEB scores
for the same models to contextualise performance without overclaiming.

### 4.2 Model Selection Rationale

We selected **BAAI/bge-m3** because it offers the best balance of retrieval
quality, efficiency, and multilingual capability among models in the 500 M–1 B
parameter range:

| Model | MTEB Retrieval Avg | Params | Dim | Max Tokens |
|---|---|---|---|---|
| BAAI/bge-m3 | ~0.599 | 568 M | 1024 | 8 192 |
| intfloat/e5-mistral-7b-instruct | ~0.569 | 7.1 B | 4096 | 32 768 |
| all-mpnet-base-v2 | ~0.501 | 110 M | 768 | 512 |
| all-MiniLM-L6-v2 | ~0.419 | 22 M | 384 | 512 |

bge-m3 supports dense, sparse, and ColBERT-style retrieval within a single
model, though this pipeline uses only its dense embeddings.  Its 8 192-token
context window comfortably accommodates even the longest EU legislative
paragraphs.

### 4.3 Contextualising Our Scores Against MTEB

Our evaluation differs from standard MTEB benchmarks in two ways:

1. **Cross-domain queries:** MTEB retrieval datasets typically have queries and
   passages from the same domain.  Our queries (sustainability recommendations)
   come from a different register than our passages (formal EU legislative text).
   This cross-domain bridge makes retrieval harder.

2. **Focused corpus:** Our evidence corpus contains 19 EU documents, far smaller
   than MTEB's web-scale collections.  This focused scope means absolute scores
   may be higher (fewer distractors), but ranking quality remains meaningful.

We therefore report our NDCG@10 alongside the MTEB retrieval average as a
contextualisation reference, not as a claim of state-of-the-art performance.

---

## 5. Retrieval Module

### 5.1 Architecture

The retrieval module follows a **strategy pattern** with a common
`BaseRetriever` interface.  Each strategy is implemented in its own file,
enabling clean ablation experiments:

| File | Class | Strategy |
|------|-------|----------|
| `retrieval/bm25_retriever.py` | `BM25Retriever` | Sparse lexical (BM25Okapi) |
| `retrieval/dense_retriever.py` | `DenseRetriever` | Dense semantic (FAISS HNSW) |
| `retrieval/hybrid_retriever.py` | `HybridRetriever` | BM25 + FAISS + RRF fusion |
| `retrieval/reranker.py` | `RerankedRetriever` | Wraps any retriever + cross-encoder |

All retrievers expose a `retrieve(query, top_k) → RetrievalResult` method that
returns an ordered list of `Chunk` objects with their scores.

### 5.2 BM25 (Sparse Lexical Baseline)

BM25Okapi (Robertson & Zaragoza, 2009) scores documents based on term frequency,
inverse document frequency, and document length normalisation.  Query and
document texts are tokenised by whitespace and lowercased.

BM25 serves as the lexical baseline.  Its strengths are exact keyword matching
and zero embedding overhead; its weaknesses are an inability to capture
paraphrases or semantic equivalences (e.g. "extended producer responsibility"
will not match "EPR" unless both forms appear).

### 5.3 Dense FAISS (Semantic Retrieval)

Query text is encoded with the same bge-m3 sentence-transformer used for
indexing.  The FAISS HNSW index returns approximate nearest neighbours by inner
product, which equals cosine similarity for L2-normalised vectors.

Dense retrieval captures semantic meaning and handles paraphrases well, but may
miss exact regulatory terminology that BM25 captures trivially.

### 5.4 Hybrid RRF (Reciprocal Rank Fusion)

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009) combines two or more
ranked lists into a single list by scoring each document as:

$$\text{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(d)}$$

where $k = 60$ is a smoothing constant that prevents top-ranked items in one
list from dominating the fused score.  Both BM25 and Dense retrieval produce
$2 \times \texttt{top\_k}$ candidates each; RRF fuses these into a single
ranking.

The RRF paper demonstrates that this unsupervised fusion consistently
outperforms individual retrieval methods across diverse tasks, and our
experiments confirm this for EU regulatory retrieval as well.

### 5.5 Cross-Encoder Reranker

After initial retrieval, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
22 M parameters) scores each (query, chunk) pair with full bidirectional
attention.  Unlike bi-encoder retrieval (which encodes query and document
independently), the cross-encoder attends to both texts jointly, enabling
fine-grained relevance judgements at the cost of higher latency.

The pipeline retrieves $2 \times \texttt{top\_k}$ candidates from the base
retriever and reranks them down to `top_k`.  This two-stage architecture
(cheap first stage → expensive reranking) is standard practice and typically
improves precision at low k values significantly.

---

## 6. RAG Alignment Classification

### 6.1 Classifier Model

**Model:** `Qwen/Qwen2.5-7B-Instruct` (Alibaba DAMO Academy)

This model was selected for the following reasons:

- **Strong instruction-following:** Qwen2.5-7B-Instruct is among the
  top-ranked 7 B models on MTEB for instruction-adherence tasks, which
  is critical for producing the structured JSON output we require.
- **Structured output:** The model reliably produces valid JSON when
  instructed to do so, reducing parsing failures.
- **Efficiency:** At 7.6 B parameters, it runs comfortably in fp16 on a
  16 GB GPU, or in 4-bit quantisation on 8 GB (with `bitsandbytes`).
- **Open-source:** Apache 2.0 license, fully transparent weights.

| Property | Value |
|----------|-------|
| Parameters | 7.6 B |
| Context length | 32 768 tokens |
| Licence | Apache 2.0 |
| VRAM (fp16) | ~15 GB |
| VRAM (4-bit) | ~5 GB |

### 6.2 Classification Procedure

For each recommendation, the classifier receives:

1. A **system prompt** (in `rag/prompts.py`) instructing it to act as a
   legal-policy analyst specialising in EU sustainability regulation, to
   base its answer **only** on the provided evidence chunks, and to output
   strict JSON.
2. A **user prompt** containing the recommendation text and the top-k
   retrieved evidence chunks (formatted with their chunk IDs, article
   references, and full text).

The model outputs a JSON object:

```json
{
  "label": "<alignment label>",
  "justification": "<evidence-based reasoning>",
  "cited_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}
```

### 6.3 Alignment Label Taxonomy

| Label | Definition |
|-------|------------|
| **Aligned** | The recommendation is directly supported by existing EU legislation. |
| **Conditional** | Alignment depends on delegated or implementing acts not yet adopted. |
| **Conflicting** | The recommendation contradicts provisions in the current framework. |
| **No explicit legal basis** | No EU provision addresses the recommendation. |
| **Beyond compliance** | The recommendation exceeds what current legislation requires. |

These labels were designed to capture the full spectrum of alignment outcomes
relevant to policymakers: from direct legal support to active contradiction.

---

## 7. LLM-as-Judge Evaluation

### 7.1 Methodology

Following Zheng et al. (2023) — *"Judging LLM-as-a-Judge with MT-Bench and
Chatbot Arena"* (NeurIPS 2023) — we implement an automated evaluation of
classification quality using a second LLM.  Three design choices are critical:

1. **Different model family.**  The judge (`mistralai/Mistral-7B-Instruct-v0.3`,
   7.2 B parameters) uses a different architecture from the classifier
   (Qwen2.5-7B-Instruct) to avoid self-evaluation bias.  Studies show that
   LLMs tend to rate their own outputs more favourably; using a different
   model family mitigates this.

2. **Multi-dimensional scoring.**  Rather than a single quality score, the
   judge evaluates three orthogonal dimensions:

   | Dimension | What it Measures | Score 5 (Best) | Score 1 (Worst) |
   |---|---|---|---|
   | Label correctness | Is the alignment label appropriate given the evidence? | Perfect label | Completely wrong |
   | Justification quality | Is the reasoning thorough and grounded in evidence? | Thorough, well-cited | Fabricated/irrelevant |
   | Evidence usage | Are the cited chunks relevant to the recommendation? | All relevant chunks cited | No relevant citations |

3. **Structured output.**  The judge produces JSON with numeric scores and
   textual reasoning, enabling both quantitative aggregation and qualitative
   inspection.

The **overall score** is the arithmetic mean of the three sub-scores.

### 7.2 Judge Model

| Property | Value |
|----------|-------|
| Model | `mistralai/Mistral-7B-Instruct-v0.3` |
| Parameters | 7.2 B |
| Context length | 32 768 tokens |
| Licence | Apache 2.0 |
| VRAM (fp16) | ~14 GB |
| VRAM (4-bit) | ~5 GB |

### 7.3 Limitations

The LLM-as-judge approach is not a substitute for human evaluation.  Known
limitations include:

- **Position bias:** Judges may favour information presented early in the prompt.
  We mitigate this by presenting evidence chunks in their retrieval-rank order.
- **Verbosity bias:** Judges may score longer justifications higher.  We do not
  control for length but report justification length alongside scores for
  transparency.
- **Ceiling effects:** On easy classifications (e.g. "Aligned" with many
  supporting provisions), scores cluster near 5.0, reducing discriminative power.

For the whitepaper recommendations (Stage 3), human evaluation is recommended
as the definitive quality assessment.

---

## 8. Evaluation Framework

### 8.1 Gold Standard

The evaluation gold standard (`data/gold_standard_doc_level/gold_standard.csv`)
contains 273 manually annotated entries from 13 academic papers.  Each entry
links a recommendation or statement to a relevant EU document at the document
level.

**Key properties:**

| Property | Value |
|---|---|
| Total entries | 273 |
| Unique EU documents | 19 |
| Academic papers | 13 |
| Evidence span types | `sentence_explicit_mention` (235), `sentence_inline_citation` (38) |
| Reference basis | `explicit_mention` (235), `citation_mapping` (38) |

Entries with `citation_mapping` reference basis were annotated based on
bibliography reference numbers in the source papers (e.g. "[39] → ESPR"),
while `explicit_mention` entries were annotated from in-text mentions
(e.g. "the WEEE directive requires…").

### 8.2 Retrieval Metrics

All retrieval metrics are computed at two granularity levels:

**Document-level:** Retrieved chunks are deduplicated to their parent EU
documents.  The ranked list for scoring purposes consists of unique document
names in first-seen order.  A document is relevant if it appears in the
gold-standard set for that query.

**Paragraph-level (chunk-level):** Each retrieved chunk is treated individually.
A chunk is relevant if its parent document is in the gold-standard set.  This
is stricter: a retriever that finds the right document but buries it among many
irrelevant chunks will have lower paragraph-level precision.

#### Metric Definitions

| Metric | Formula | Interpretation |
|---|---|---|
| Hit@k | $\mathbb{1}[\|\text{rel} \cap \text{ret@k}\| > 0]$ | Binary: did we find *anything* relevant? |
| Recall@k | $\frac{\|\text{rel} \cap \text{ret@k}\|}{\|\text{rel}\|}$ | Coverage: what fraction of relevant items did we find? |
| Precision@k | $\frac{\|\text{rel} \cap \text{ret@k}\|}{k}$ | Signal-to-noise: what fraction of top-k is relevant? |
| MRR | $\frac{1}{\text{rank of first relevant}}$ | Speed: how quickly is the first relevant item surfaced? |
| MAP | $\frac{1}{\|\text{rel}\|} \sum_r \text{Prec@r} \cdot \text{rel}(r)$ | Overall ranking quality (position-sensitive) |
| NDCG@k | $\frac{\text{DCG@k}}{\text{IDCG@k}}$ | Ranking quality with logarithmic discount for depth |
| Mean Rank (MR) | $\frac{1}{n}\sum_q \text{rank}_q$ | Average position of first relevant item (lower is better) |

#### Metric Equivalences When |relevant| = 1

A distinctive property of our gold standard is that **each recommendation
maps to exactly one EU document**.  This has important implications for
metric interpretation:

- **Recall@k ≡ Hit@k** — since there is only one relevant document,
  finding it means 100 % recall and missing it means 0 %.  The two
  metrics are numerically identical.
- **Precision@k = Hit@k / k** — a deterministic function of Hit@k,
  adding no new information.
- **MAP ≡ MRR** — Average Precision with a single relevant item reduces
  to the reciprocal of its rank, which is exactly MRR.

Consequently, the **primary metrics** we highlight in all tables are:

| Metric | Why Primary |
|---|---|
| **Hit@k** | Binary success — did the pipeline find the right EU document? |
| **MRR** | Ranking speed — how high is the correct document in the list? |
| **NDCG@k** | Ranking quality — the only metric with graded position discount |
| **Mean Rank** | Intuitive complement to MRR — "on average, the correct document appears at position X" |

The remaining metrics (Recall, Precision, MAP) are still computed and
reported for completeness and for comparability with other studies that
use them, but readers should note the equivalences above when
interpreting the results.

### 8.3 Classification Metrics

When gold-standard alignment labels are available, we compute:

| Metric | Description |
|---|---|
| Accuracy | Fraction of correct predictions |
| Macro-F1 | Unweighted mean of per-class F1 (treats rare labels equally) |
| Weighted-F1 | Support-weighted mean of per-class F1 |
| Cohen's κ | Agreement beyond chance (robust to class imbalance) |
| Per-class P/R/F1 | Precision, recall, F1 for each alignment label |
| Confusion matrix | True vs predicted label counts |

### 8.4 LLM-as-Judge Metrics

Each classification is scored on three dimensions (1–5).  We report:

- Mean, standard deviation, and 95 % bootstrap CI for each dimension
- Score distributions (histograms, violin plots)
- Mean scores broken down by predicted alignment label
- Correlation between sub-scores (label, justification, evidence)
- Fraction of classifications meeting an acceptability threshold (≥ 3.0)

---

## 9. Publishability Considerations

### 9.1 Statistical Rigour

To ensure the reported results meet the standards of peer-reviewed venues:

1. **Bootstrap confidence intervals** (95 %, 10 000 resamples, percentile
   method) are computed for all primary retrieval metrics.  Per-query score
   vectors are resampled with replacement, and the 2.5th and 97.5th
   percentiles of the bootstrap distribution define the interval.

2. **Paired permutation tests** (two-sided, 10 000 permutations) assess
   whether observed differences between retrieval configurations are
   statistically significant.  For each permutation, the signs of per-query
   score differences are randomly flipped, and the proportion of permutations
   where the absolute mean difference exceeds the observed difference gives
   the p-value.  Differences with p < 0.05 are flagged.

3. **Effect sizes** are reported as absolute deltas between retriever pairs
   alongside p-values, since statistical significance does not imply
   practical significance.

### 9.2 Error Analysis

Retrieval failures are categorised for the best-performing configuration:

| Category | Definition |
|---|---|
| **Success** | All relevant documents found in top-k |
| **Partial** | Some relevant documents found, others missed |
| **Late hit** | Correct document found but only at rank > k |
| **Complete miss** | Correct document not found even in extended top-20 |

This categorisation reveals *why* retrieval fails (vocabulary mismatch, semantic
gap, ambiguous references) and informs targeted improvements.

### 9.3 Human Evaluation of Whitepaper Results

For the unlabelled whitepaper recommendations (Stage 3), the exported CSV
includes empty `human_label` and `human_notes` columns.  We recommend:

- **Two independent evaluators** assess each recommendation to enable
  inter-annotator agreement (Cohen's κ or Krippendorff's α).
- Evaluators see: the recommendation, the retrieved evidence, the LLM's
  label and justification.
- They assign their own alignment label and note any disagreements.
- Agreement metrics are computed and reported to establish the reliability
  of the ground truth.

---

## 10. Reproducibility

### 10.1 Environment Setup

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\Activate.ps1      # Windows

pip install -r requirements.txt
```

### 10.2 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB (4-bit quantised) | 16+ GB (fp16) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB (models) | 40 GB |

### 10.3 Running the Pipeline

```bash
# 1. Build indices from evidence
python pipeline.py build -i outputs/evidence.csv -m bge-m3

# 2. Evaluate retrieval on gold standard
python pipeline.py evaluate --gold data/gold_standard_doc_level/gold_standard.csv

# 3. Run external benchmark evaluation
python benchmarks/generate_benchmark.py          # generate benchmark JSON
python pipeline.py benchmark -i benchmarks/gold_standard_benchmark.json

# 4. Run whitepaper recommendations (retrieve + classify)
python pipeline.py whitepaper -o outputs/whitepaper_classified.csv

# 5. Run whitepaper retrieval only (no classification / no GPU required)
python pipeline.py whitepaper --retrieve-only -o outputs/whitepaper_retrieval.csv

# 6. Run full pipeline (retrieve + classify + evaluate)
python pipeline.py run -i outputs/recommendations.csv \
                   --gold data/gold_standard_doc_level/gold_standard.csv
```

### 10.4 Running Notebooks

```bash
cd notebooks
jupyter notebook
```

Execute in order:

1. `01_retrieval_analysis.ipynb` — Retrieval strategy comparison (document +
   paragraph level, bootstrap CIs, statistical significance, error analysis)
2. `02_rag_evaluation.ipynb` — Classification + LLM-as-judge
3. `03_evaluation_metrics.ipynb` — Consolidated evaluation breakdown
4. `04_whitepaper_evaluation.ipynb` — Whitepaper recommendations (retrieval +
   classification, export for human evaluation)

### 10.5 Random Seeds

All stochastic operations use fixed random seeds for reproducibility:

- Bootstrap CI: `rng_seed=42`
- Paired permutation test: `rng_seed=42`
- LLM temperature: `0.0` (deterministic decoding)

---

## 11. File Reference

### Core Modules

| File | Purpose |
|------|---------|
| `config.py` | Centralised configuration (paths, models, hyperparameters) |
| `data_models.py` | Domain dataclasses (Chunk, Recommendation, metrics) |
| `embedding_indexing.py` | Embedding generation, FAISS/BM25 index building |
| `pipeline.py` | CLI orchestration (build, evaluate, classify, run) |
| `main.py` | Entry point |

### Retrieval Package

| File | Purpose |
|------|---------|
| `retrieval/base_retriever.py` | Abstract base class (`BaseRetriever`) |
| `retrieval/bm25_retriever.py` | BM25 sparse retriever |
| `retrieval/dense_retriever.py` | FAISS dense retriever |
| `retrieval/hybrid_retriever.py` | RRF fusion retriever |
| `retrieval/reranker.py` | Cross-encoder reranker + wrapper |
| `retrieval/chunking_evidence.py` | HTML → paragraph-level chunks |
| `retrieval/chunking_recommendations.py` | TXT → atomic recommendations |

### RAG Package

| File | Purpose |
|------|---------|
| `rag/classifier.py` | Open-source LLM alignment classifier |
| `rag/llm_judge.py` | LLM-as-judge evaluation |
| `rag/prompts.py` | All prompt templates |

### Evaluation Package

| File | Purpose |
|------|---------|
| `evaluation/evaluation.py` | Gold-standard loading, document- and paragraph-level evaluation, whitepaper loader |
| `evaluation/metrics.py` | Metric computation (retrieval + classification), bootstrap CI, paired permutation test |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_retrieval_analysis.ipynb` | BM25 vs FAISS vs Hybrid ± Reranker (doc + para level, CIs, significance, error analysis) |
| `02_rag_evaluation.ipynb` | Classification + LLM-as-judge |
| `03_evaluation_metrics.ipynb` | Consolidated evaluation breakdown |
| `04_whitepaper_evaluation.ipynb` | Whitepaper recommendations → retrieval + classification → human evaluation CSV |

---

## 12. References

1. Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
   *Reciprocal Rank Fusion outperforms Condorcet and individual Rank
   Learning Methods.* SIGIR '09.

2. Zheng, L., Chiang, W.-L., et al. (2023).
   *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
   NeurIPS 2023.

3. Muennighoff, N., et al. (2023).
   *MTEB: Massive Text Embedding Benchmark.* EACL 2023.

4. Chen, J., et al. (2024).
   *BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
   Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.*
   arXiv:2402.03216.

5. Robertson, S. & Zaragoza, H. (2009).
   *The Probabilistic Relevance Framework: BM25 and Beyond.*
   Foundations and Trends in Information Retrieval, 3(4), 333–389.

6. Efron, B. & Tibshirani, R. J. (1993).
   *An Introduction to the Bootstrap.* Chapman & Hall/CRC.

7. Yang, Z., et al. (2024).
   *Qwen2.5 Technical Report.* arXiv:2412.15115.

8. Jiang, A. Q., et al. (2023).
   *Mistral 7B.* arXiv:2310.06825.
