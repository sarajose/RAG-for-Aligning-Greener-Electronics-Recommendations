
# RAG-for-Aligning-Greener-Electronics-Recommendations

**What is it?**

This project provides a fully open-source pipeline to automatically map sustainability recommendations for the electronics sector to relevant EU legislation. It retrieves the most relevant legal provisions, classifies the degree of alignment using LLMs, and evaluates results against a gold standard and external benchmarks. All models run locally for reproducibility and privacy.

---

## Models Tested

- **Embedding models:**
	- `BAAI/bge-m3` (default, 1024-dim, top MTEB performer)
	- `all-mpnet-base-v2` (`mpnet`)
	- `all-MiniLM-L6-v2` (`minilm`)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Alignment classifier:** `Qwen/Qwen2.5-7B-Instruct`
- **LLM-as-Judge:** `mistralai/Mistral-7B-Instruct-v0.3`

---

## Evaluation

- **Gold-standard document-level retrieval:** Can the system find the correct EU document for each recommendation?
- **Paragraph-level retrieval:** Does it surface the right provisions, not just the right document?
- **Whitepaper export:** Produces outputs for 48 real recommendations for human review.
- **External benchmark:** MTEB LegalBench chunk-level retrieval.
- **Metrics:** Hit@k, Recall@k, Precision@k, MRR, MAP, NDCG, F1, Cohen's κ, bootstrap CIs, permutation tests.

---

## Usage

### 1. Environment Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. (Optional) Download Models

```powershell
python pipeline.py download-models
```

### 3. Build Indices (per model)

```powershell
python main.py build -i outputs/evidence.csv -m bge-m3
python main.py build -i outputs/evidence.csv -m mpnet
python main.py build -i outputs/evidence.csv -m minilm
```

### 4. Unified Evaluation

```powershell
python pipeline.py unified-eval --models bge-m3 mpnet minilm
```

### 5. Visualize Results

Open `notebooks/05_eval_visualizations_only.ipynb` for plots and inspection.

---

## File Structure

```
├── config.py                  # All paths, models, and hyperparameters
├── data_models.py             # Shared dataclasses: Chunk, Recommendation, metrics
├── embedding_indexing.py      # Embed chunks, build FAISS + BM25 indices
├── pipeline.py                # Unified CLI: build/evaluate/classify/run/unified-eval
├── main.py                    # Entry point (delegates to pipeline.py)
│
├── retrieval/
│   ├── base_retriever.py           # BaseRetriever ABC (retrieve interface)
│   ├── bm25_retriever.py           # BM25Retriever — keyword search
│   ├── dense_retriever.py          # DenseRetriever — FAISS semantic search
│   ├── hybrid_retriever.py         # HybridRetriever — BM25 + FAISS + RRF fusion
│   ├── reranker.py                 # Reranker (cross-encoder) + RerankedRetriever
│   ├── chunking_evidence.py        # EUR-Lex HTML → paragraph-level chunk CSV
│   ├── chunking_recommendations.py # TXT → atomic recommendation CSV
│   └── retrieval.py                # Low-level search utilities
│
├── rag/
│   ├── classifier.py          # AlignmentClassifier (Qwen2.5-7B)
│   ├── llm_judge.py           # LLMJudge (Mistral-7B)
│   └── prompts.py             # Prompt templates
│
├── evaluation/
│   ├── evaluation.py          # Gold/para-level evaluation
│   └── metrics.py             # Metrics, bootstrap CI, permutation test
│
├── notebooks/
│   └── 05_eval_visualizations_only.ipynb  # Unified evaluation visualizations
│
├── docs/
│   ├── PIPELINE.md            # Pipeline walkthrough and reference
│   └── METHODOLOGY.md         # Detailed methodology
│
├── data/
│   ├── evidence/                              # EUR-Lex HTML documents
│   ├── gold_standard_doc_level/
│   │   └── gold_standard.csv                 # 273 annotated recommendation→doc pairs
│   └── recommendations_whitepaper/
│       └── recommendations_empty.csv         # 48 whitepaper recommendations
│
└── outputs/
		├── evidence.csv                           # Chunked evidence
		└── indices/
				├── bge-m3_faiss.index                 # FAISS HNSW index
				├── bge-m3_bm25.pkl                    # BM25Okapi index
				└── bge-m3_chunks.pkl                  # Chunk list (aligned with index)
```

---

**Unified results are written to `outputs/eval_unified/`.**