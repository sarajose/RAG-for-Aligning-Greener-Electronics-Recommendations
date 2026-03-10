"""
Hybrid retriever combining BM25 + FAISS via Reciprocal Rank Fusion.

Implements the RRF algorithm from "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (Cormack et al., 2009).
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import DEFAULT_MODEL_KEY, DEFAULT_TOP_K, RRF_K
from data_models import Chunk, RetrievalResult
from embedding_indexing import (
    get_embed_model,
    embed_texts,
    load_indices,
    tokenize,
)
from retrieval.base_retriever import BaseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists using RRF.

    Parameters
    ----------
    ranked_lists : list[list[int]]
        Each inner list contains chunk indices ordered by relevance.
    k : int
        Smoothing constant (default 60).

    Returns
    -------
    list[tuple[int, float]]
        ``(chunk_index, fused_score)`` sorted descending.
    """
    fused: dict[int, float] = {}
    for rlist in ranked_lists:
        for rank, idx in enumerate(rlist, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever(BaseRetriever):
    """BM25 + Dense FAISS with Reciprocal Rank Fusion."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25: BM25Okapi,
        chunks: list[Chunk],
        embed_model: SentenceTransformer,
    ) -> None:
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.chunks = chunks
        self.embed_model = embed_model

    @property
    def name(self) -> str:
        return "Hybrid (BM25 + FAISS + RRF)"

    @classmethod
    def from_disk(cls, model_key: str = DEFAULT_MODEL_KEY) -> "HybridRetriever":
        """Load pre-built indices and embedding model from disk."""
        fi, bm25, chunks = load_indices(model_key)
        embed_model = get_embed_model(model_key)
        return cls(fi, bm25, chunks, embed_model)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        # BM25 sparse search (2x candidates for fusion)
        tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][: top_k * 2]

        # Dense FAISS search (2x candidates for fusion)
        q_emb = embed_texts([query], self.embed_model, show_progress=False)
        _, dense_idx = self.faiss_index.search(
            q_emb.astype(np.float32), top_k * 2,
        )
        dense_idx = dense_idx[0]

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            [bm25_idx.tolist(), dense_idx.tolist()],
        )
        final = fused[:top_k]

        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[idx] for idx, _ in final],
            scores=[score for _, score in final],
        )
