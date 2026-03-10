"""
BM25 sparse-lexical retriever (baseline).

Uses BM25Okapi for term-frequency–based document ranking.
This serves as the lexical baseline against which dense and
hybrid approaches are compared.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from config import DEFAULT_MODEL_KEY, DEFAULT_TOP_K
from data_models import Chunk, RetrievalResult
from embedding_indexing import load_indices, tokenize
from retrieval.base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    """Sparse BM25 retriever."""

    def __init__(self, bm25: BM25Okapi, chunks: list[Chunk]) -> None:
        self.bm25 = bm25
        self.chunks = chunks

    @property
    def name(self) -> str:
        return "BM25"

    @classmethod
    def from_disk(cls, model_key: str = DEFAULT_MODEL_KEY) -> "BM25Retriever":
        """Load pre-built BM25 index and chunk list from disk."""
        _, bm25, chunks = load_indices(model_key)
        return cls(bm25, chunks)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[i] for i in top_idx],
            scores=[float(scores[i]) for i in top_idx],
        )
