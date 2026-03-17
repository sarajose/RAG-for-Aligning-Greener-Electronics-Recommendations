"""
Hybrid retrieval engine for the policy-alignment pipeline.

Combines dense vector search (FAISS), sparse lexical search (BM25Okapi),
and optional cross-encoder reranking through Reciprocal Rank Fusion (RRF).

Architecture::

    Query ──┬──▶ BM25       ──┐
            │                  ├──▶ RRF fusion ──▶ (optional) Rerank ──▶ Top-K
            └──▶ Dense/FAISS ─┘

Typical usage::

    from retrieval import HybridRetriever

    retriever = HybridRetriever.from_disk("bge-m3")
    result = retriever.retrieve("ban lead in solder", top_k=10)
    for chunk, score in zip(result.ranked_chunks, result.scores):
        print(chunk.article, score)
"""

import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

from config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP,
    RERANKER_MODEL,
    RRF_K,
)
from data_models import Chunk, RetrievalResult
from embedding_indexing import (
    get_embed_model,
    embed_texts,
    load_indices,
    tokenize,
)

# Low-level search primitives

def search_faiss(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Query the FAISS dense-vector index.

    Parameters
    ----------
    index : faiss.Index
        Pre-built FAISS index.
    query_embedding : np.ndarray
        ``(1, D)`` query vector (L2-normalised).
    k : int
        Number of nearest neighbours.

    Returns
    -------
    scores : np.ndarray
        Shape ``(k,)`` cosine similarities.
    indices : np.ndarray
        Shape ``(k,)`` chunk-list positions.
    """
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    return scores[0], indices[0]


def search_bm25(
    bm25: BM25Okapi,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Query the BM25 sparse-lexical index.

    Parameters
    ----------
    bm25 : BM25Okapi
        Pre-built BM25 index.
    query : str
        Free-text query string.
    k : int
        Number of results.

    Returns
    -------
    scores : np.ndarray
        Shape ``(k,)`` BM25 relevance scores.
    indices : np.ndarray
        Shape ``(k,)`` chunk-list positions.
    """
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return scores[top_idx], top_idx


# Reciprocal Rank Fusion

def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using RRF (Cormack et al., 2009).

    Parameters
    ----------
    ranked_lists : list[list[int]]
        Each inner list contains chunk indices ordered by relevance.
    k : int
        Smoothing constant (default 60).

    Returns
    -------
    list[tuple[int, float]]
        ``(chunk_index, fused_score)`` pairs sorted descending by score.
    """
    fused: dict[int, float] = {}
    for rlist in ranked_lists:
        for rank, idx in enumerate(rlist, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# Cross-encoder reranking

def rerank(
    query: str,
    chunks: list[Chunk],
    candidate_indices: list[int],
    top_k: int = DEFAULT_RERANK_TOP,
    model_name: str = RERANKER_MODEL,
) -> list[tuple[int, float]]:
    """Re-score candidate chunks with a cross-encoder.

    Parameters
    ----------
    query : str
        Recommendation / question text.
    chunks : list[Chunk]
        Full chunk list (indices reference into this).
    candidate_indices : list[int]
        Chunk indices from the initial (hybrid) ranking.
    top_k : int
        How many results to keep after reranking.
    model_name : str
        HuggingFace cross-encoder model identifier.

    Returns
    -------
    list[tuple[int, float]]
        ``(chunk_index, cross_encoder_score)`` — best first.
    """
    ce = CrossEncoder(model_name)
    pairs = [(query, chunks[i].text) for i in candidate_indices]
    scores = ce.predict(pairs)
    order = np.argsort(scores)[::-1][:top_k]
    return [(candidate_indices[i], float(scores[i])) for i in order]


# High-level retriever

class HybridRetriever:
    """BM25 + dense FAISS retrieval with RRF fusion and optional reranking.

    Parameters
    ----------
    faiss_index : faiss.Index
        Pre-built dense-vector index.
    bm25 : BM25Okapi
        Pre-built sparse-lexical index.
    chunks : list[Chunk]
        Ordered chunk list matching index positions.
    embed_model : SentenceTransformer
        Model used for query encoding (must match the index).
    use_reranker : bool
        Whether to apply cross-encoder reranking on retrieval.
    """

    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
        chunks: list[Chunk],
        embed_model: SentenceTransformer,
        use_reranker: bool = True,
    ) -> None:
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.chunks = chunks
        self.embed_model = embed_model
        self.use_reranker = use_reranker

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_disk(
        cls,
        model_key: str = DEFAULT_MODEL_KEY,
        use_reranker: bool = True,
    ) -> "HybridRetriever":
        """Load indices from disk and return a ready-to-query retriever.

        Parameters
        ----------
        model_key : str
            Short key identifying the embedding model (and index files).
        use_reranker : bool
            Enable cross-encoder reranking.
        """
        print(f"[retrieval] Loading indices ({model_key})")
        fi, bm, chunks = load_indices(model_key)
        embed_model = get_embed_model(model_key)
        return cls(fi, bm, chunks, embed_model, use_reranker)

    # ── Core retrieval method ────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rerank_top: int = DEFAULT_RERANK_TOP,
    ) -> RetrievalResult:
        """Retrieve and optionally rerank chunks for *query*.

        Pipeline stages:

        1. BM25 sparse search → 2×top_k candidates
        2. Dense FAISS search → 2×top_k candidates
        3. RRF fusion          → top_k candidates
        4. Cross-encoder rerank (if enabled) → rerank_top results

        Parameters
        ----------
        query : str
            Free-text recommendation or question.
        top_k : int
            Number of hybrid candidates before reranking.
        rerank_top : int
            Final number of results after reranking.

        Returns
        -------
        RetrievalResult
        """
        # 1 & 2 — Sparse + Dense
        _, bm25_idx = search_bm25(self.bm25, query, k=top_k * 2)
        q_emb = embed_texts(
            [query], self.embed_model, show_progress=False, is_query=True,
        )
        _, dense_idx = search_faiss(self.faiss_index, q_emb, k=top_k * 2)

        # 3 — RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_idx.tolist(), dense_idx.tolist()]
        )
        hybrid_indices = [idx for idx, _ in fused[:top_k]]

        # 4 — Rerank (optional)
        if self.use_reranker:
            reranked = rerank(
                query, self.chunks, hybrid_indices, top_k=rerank_top,
            )
            final_indices = [idx for idx, _ in reranked]
            final_scores = [sc for _, sc in reranked]
        else:
            final_indices = hybrid_indices[:rerank_top]
            final_scores = [sc for _, sc in fused[:rerank_top]]

        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[i] for i in final_indices],
            scores=final_scores,
        )
