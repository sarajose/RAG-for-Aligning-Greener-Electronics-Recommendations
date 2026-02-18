"""
Embedding, Indexing, and Hybrid Retrieval for Legal Provisions

Reads chunked CSV from outputs/, embeds text with SOTA models, and builds:
  - Dense vector index (FAISS HNSW for ANN search)
  - Sparse BM25 index (lexical retrieval)
  - Hybrid retrieval (BM25 + dense fusion)
  - Cross-encoder reranker for precision

Supports multiple embedding models:
  - BAAI/bge-m3 (multilingual, 1024d, best for legal/technical text)
  - intfloat/e5-mistral-7b-instruct (7B params, instruction-tuned)
  - sentence-transformers/all-mpnet-base-v2 (768d, balanced)
  - sentence-transformers/all-MiniLM-L6-v2 (384d, fast)

Usage:
  # Build indices from CSV
  python embedding_indexing.py build -i outputs/evidence.csv -m bge-m3

  # Query hybrid retrieval
  python embedding_indexing.py query -q "What are the obligations for manufacturers?" -k 10

  # Evaluate on test set
  python embedding_indexing.py evaluate -t test_queries.json
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# CONFIGURATION

EMBEDDING_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_DIR = Path("indices")
INDEX_DIR.mkdir(exist_ok=True)


@dataclass
class Chunk:
    """A single legal provision chunk."""
    id: str
    document: str
    source_file: str
    version: str
    chapter: str
    article: str
    article_subtitle: str
    paragraph: str
    char_offset: int
    text: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document": self.document,
            "source_file": self.source_file,
            "version": self.version,
            "chapter": self.chapter,
            "article": self.article,
            "article_subtitle": self.article_subtitle,
            "paragraph": self.paragraph,
            "char_offset": self.char_offset,
            "text": self.text,
        }

# 2. DATA LOADING

def load_chunks(csv_path: Path) -> list[Chunk]:
    """Load chunks from CSV."""
    chunks: list[Chunk] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunks.append(Chunk(
                id=row["id"],
                document=row["document"],
                source_file=row["source_file"],
                version=row["version"],
                chapter=row["chapter"],
                article=row["article"],
                article_subtitle=row["article_subtitle"],
                paragraph=row["paragraph"],
                char_offset=int(row["char_offset"]) if row["char_offset"] else 0,
                text=row["text"],
            ))
    return chunks


# 3. EMBEDDING GENERATION

def get_embed_model(model_key: str) -> SentenceTransformer:
    """Load a sentence-transformer embedding model."""
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(EMBEDDING_MODELS.keys())}")
    model_name = EMBEDDING_MODELS[model_key]
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str], model: SentenceTransformer, batch_size: int = 32, show_progress: bool = True
) -> np.ndarray:
    """Encode texts into dense vectors."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    return embeddings


# 4. FAISS INDEX (Dense Vector Search)

def build_faiss_index(embeddings: np.ndarray, use_hnsw: bool = True) -> faiss.Index:
    """Build a FAISS index for approximate nearest neighbor search.
    
    Args:
        embeddings: (N, D) array of L2-normalized vectors
        use_hnsw: If True, use HNSW (fast, accurate). If False, use Flat (exact, slower).
    """
    dim = embeddings.shape[1]
    if use_hnsw:
        # HNSW: Hierarchical Navigable Small World graphs
        # M = number of bi-directional links (16-64 typical)
        # efConstruction = depth of search during construction (higher = better quality, slower)
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16  # search depth at query time (tune for recall/speed)
    else:
        # Flat index: exact search (baseline)
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vectors
    
    index.add(embeddings.astype(np.float32))
    return index


def search_faiss(
    index: faiss.Index, query_embedding: np.ndarray, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index.
    
    Returns:
        scores: (k,) cosine similarities
        indices: (k,) chunk indices
    """
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    return scores[0], indices[0]

# 5. BM25 INDEX (Sparse Lexical Search)

def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    return text.lower().split()


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """Build a BM25 index."""
    tokenized = [tokenize(t) for t in texts]
    return BM25Okapi(tokenized)


def search_bm25(bm25: BM25Okapi, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Search BM25 index.
    
    Returns:
        scores: (k,) BM25 scores
        indices: (k,) chunk indices
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    # Get top-k indices
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_idx]
    return top_k_scores, top_k_idx


# 6. HYBRID RETRIEVAL (Reciprocal Rank Fusion)

def reciprocal_rank_fusion(
    results: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        results: list of ranked lists (each list is chunk indices)
        k: RRF parameter (default 60)
    
    Returns:
        List of (chunk_idx, fused_score) sorted by score descending
    """
    scores: dict[int, float] = {}
    for rank_list in results:
        for rank, idx in enumerate(rank_list, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    bm25: BM25Okapi,
    faiss_index: faiss.Index,
    embed_model: SentenceTransformer,
    k: int = 10,
    bm25_weight: float = 0.5,
) -> list[int]:
    """Hybrid retrieval: BM25 + dense vector search with RRF fusion.
    
    Args:
        query: query text
        bm25: BM25 index
        faiss_index: FAISS dense index
        embed_model: embedding model
        k: number of results to return
        bm25_weight: weight for BM25 (not used in RRF, but kept for future weighted fusion)
    
    Returns:
        List of chunk indices (fused top-k)
    """
    # BM25 retrieval
    _, bm25_indices = search_bm25(bm25, query, k=k*2)
    
    # Dense retrieval
    query_emb = embed_texts([query], embed_model, show_progress=False)
    _, dense_indices = search_faiss(faiss_index, query_emb, k=k*2)
    
    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([bm25_indices.tolist(), dense_indices.tolist()])
    return [idx for idx, score in fused[:k]]


# 7. RERANKING (Cross-Encoder)

def rerank(
    query: str, chunks: list[Chunk], indices: list[int], top_k: int = 5
) -> list[int]:
    """Rerank top-k results using a cross-encoder.
    
    Args:
        query: query text
        chunks: all chunks
        indices: initial ranking (chunk indices)
        top_k: number of results to return after reranking
    
    Returns:
        Reranked chunk indices (top_k)
    """
    reranker = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, chunks[idx].text) for idx in indices]
    scores = reranker.predict(pairs)
    reranked_idx = np.argsort(scores)[::-1][:top_k]
    return [indices[i] for i in reranked_idx]


# 8. INDEX PERSISTENCE

def save_indices(
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    model_key: str,
) -> None:
    """Save indices and chunks to disk."""
    prefix = INDEX_DIR / model_key
    faiss.write_index(faiss_index, str(prefix) + "_faiss.index")
    with open(prefix.with_name(prefix.name + "_bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(prefix.with_name(prefix.name + "_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved indices to {INDEX_DIR}/")


def load_indices(model_key: str) -> tuple[faiss.Index, BM25Okapi, list[Chunk]]:
    """Load indices and chunks from disk."""
    prefix = INDEX_DIR / model_key
    faiss_index = faiss.read_index(str(prefix) + "_faiss.index")
    with open(prefix.with_name(prefix.name + "_bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(prefix.with_name(prefix.name + "_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return faiss_index, bm25, chunks


# 9. BUILD PIPELINE

def build_cmd(csv_path: Path, model_key: str = "bge-m3") -> None:
    """Build all indices from a CSV file."""
    print(f"Loading chunks from {csv_path}...")
    chunks = load_chunks(csv_path)
    texts = [c.text for c in chunks]
    print(f"Loaded {len(chunks)} chunks")

    # Embed
    embed_model = get_embed_model(model_key)
    print("Generating embeddings...")
    embeddings = embed_texts(texts, embed_model)
    print(f"Embeddings shape: {embeddings.shape}")

    # FAISS index
    print("Building FAISS HNSW index")
    faiss_index = build_faiss_index(embeddings, use_hnsw=True)

    # BM25 index
    print("Building BM25 index")
    bm25 = build_bm25_index(texts)

    # Save
    save_indices(faiss_index, bm25, chunks, model_key)
    print("Build complete")


# 10. QUERY PIPELINE

def query_cmd(
    query: str,
    model_key: str = "bge-m3",
    k: int = 10,
    rerank_top: int = 5,
    use_reranker: bool = True,
    output_file: Optional[Path] = None,
) -> None:
    """Query the hybrid retrieval system and optionally save results to file."""
    print(f"Loading indices ({model_key})")
    faiss_index, bm25, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)

    print(f"\nQuery: {query}")

    # Hybrid search
    hybrid_indices = hybrid_search(query, bm25, faiss_index, embed_model, k=k)

    # Rerank
    if use_reranker:
        print(f"Reranking top {rerank_top}...")
        final_indices = rerank(query, chunks, hybrid_indices, top_k=rerank_top)
    else:
        final_indices = hybrid_indices[:rerank_top]

    # Display results
    results = []
    for rank, idx in enumerate(final_indices, start=1):
        c = chunks[idx]
        print(f"\n[{rank}] {c.document} | {c.article} | Para {c.paragraph}")
        print(f"    {c.text[:200]}...")
        results.append({
            "query": query,
            "rank": rank,
            "chunk_id": c.id,
            "document": c.document,
            "article": c.article,
            "paragraph": c.paragraph,
            "text": c.text,
        })

    # Save to file if requested
    if output_file:
        output_file = Path(output_file)
        file_exists = output_file.exists()
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_file}")


def query_batch_cmd(
    recommendations_csv: Path,
    model_key: str = "bge-m3",
    k: int = 10,
    rerank_top: int = 5,
    use_reranker: bool = True,
    output_file: Optional[Path] = None,
) -> None:
    """Query the hybrid retrieval system with multiple recommendations from CSV."""
    print(f"Loading indices ({model_key})")
    faiss_index, bm25, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)

    print(f"\nLoading recommendations from {recommendations_csv}")
    recommendations = []
    with open(recommendations_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            recommendations.append(row)
    
    print(f"Loaded {len(recommendations)} recommendations\n")

    all_results = []

    for i, rec in enumerate(recommendations, start=1):
        query = rec.get("recommendation", "")
        if not query.strip():
            print(f"[{i}/{len(recommendations)}] Skipping empty recommendation")
            continue

        print(f"[{i}/{len(recommendations)}] Query: {query[:80]}...")

        # Hybrid search
        hybrid_indices = hybrid_search(query, bm25, faiss_index, embed_model, k=k)

        # Rerank
        if use_reranker:
            final_indices = rerank(query, chunks, hybrid_indices, top_k=rerank_top)
        else:
            final_indices = hybrid_indices[:rerank_top]

        # Collect results
        for rank, idx in enumerate(final_indices, start=1):
            c = chunks[idx]
            result = {
                "section": rec.get("section", ""),
                "subsection": rec.get("subsection", ""),
                "title": rec.get("title", ""),
                "recommendation": query,
                "rank": rank,
                "chunk_id": c.id,
                "document": c.document,
                "article": c.article,
                "paragraph": c.paragraph,
                "text": c.text,
            }
            all_results.append(result)

    # Save all results to file
    if output_file and all_results:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {output_file} ({len(all_results)} total matches)")
    elif all_results:
        print(f"\nGenerated {len(all_results)} total matches (no output file specified)")


# 11. EVALUATION

def evaluate_cmd(test_path: Path, model_key: str = "bge-m3", k: int = 10) -> None:
    """Evaluate retrieval on a test set.
    
    Test file format (JSON):
    [
      {"query": "...", "relevant_ids": ["chunk_id1", "chunk_id2", ...]},
      ...
    ]
    """
    print(f"Loading test set from {test_path}...")
    with open(test_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    faiss_index, bm25, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)
    id_to_idx = {c.id: i for i, c in enumerate(chunks)}

    recalls = []
    mrrs = []

    for case in test_cases:
        query = case["query"]
        relevant_ids = set(case["relevant_ids"])
        
        # Retrieve
        hybrid_indices = hybrid_search(query, bm25, faiss_index, embed_model, k=k)
        retrieved_ids = [chunks[i].id for i in hybrid_indices]

        # Compute metrics
        hits = [rid in relevant_ids for rid in retrieved_ids]
        recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0
        
        # MRR: first relevant position
        mrr = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                mrr = 1.0 / rank
                break
        
        recalls.append(recall)
        mrrs.append(mrr)

    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mrrs)

    print("\nEvaluation Results:")
    print(f"  Recall{k}: {avg_recall:.3f}")
    print(f"  MRR{k}:    {avg_mrr:.3f}")


# 12. CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embedding, indexing, and hybrid retrieval for legal provisions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_p = subparsers.add_parser("build", help="Build indices from CSV")
    build_p.add_argument("-i", "--input", required=True, type=Path, help="Input CSV path")
    build_p.add_argument(
        "-m", "--model", default="bge-m3",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model to use"
    )

    # Query command
    query_p = subparsers.add_parser("query", help="Query the retrieval system")
    query_p.add_argument("-q", "--query", required=True, help="Query text")
    query_p.add_argument("-m", "--model", default="bge-m3", choices=list(EMBEDDING_MODELS.keys()))
    query_p.add_argument("-k", "--top-k", type=int, default=10, help="Number of results")
    query_p.add_argument("-o", "--output", type=Path, default=None, help="Output CSV file to save results")
    query_p.add_argument("--no-rerank", action="store_true", help="Skip reranking")

    # Batch query command
    batch_p = subparsers.add_parser("query-batch", help="Query with multiple recommendations from CSV")
    batch_p.add_argument("-i", "--input", required=True, type=Path, help="Input CSV file with recommendations (columns: section, subsection, title, recommendation)")
    batch_p.add_argument("-o", "--output", required=True, type=Path, help="Output CSV file to save results")
    batch_p.add_argument("-m", "--model", default="bge-m3", choices=list(EMBEDDING_MODELS.keys()))
    batch_p.add_argument("-k", "--top-k", type=int, default=10, help="Number of results per recommendation")
    batch_p.add_argument("--no-rerank", action="store_true", help="Skip reranking")

    # Evaluate command
    eval_p = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_p.add_argument("-t", "--test", required=True, type=Path, help="Test JSON file")
    eval_p.add_argument("-m", "--model", default="bge-m3", choices=list(EMBEDDING_MODELS.keys()))
    eval_p.add_argument("-k", "--top-k", type=int, default=10)

    args = parser.parse_args()

    if args.command == "build":
        build_cmd(args.input, args.model)
    elif args.command == "query":
        query_cmd(args.query, args.model, args.top_k, rerank_top=5, use_reranker=not args.no_rerank, output_file=args.output)
    elif args.command == "query-batch":
        query_batch_cmd(args.input, args.model, args.top_k, rerank_top=args.top_k, use_reranker=not args.no_rerank, output_file=args.output)
    elif args.command == "evaluate":
        evaluate_cmd(args.test, args.model, args.top_k)


if __name__ == "__main__":
    main()
