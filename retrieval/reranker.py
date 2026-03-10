"""
Cross-encoder reranking module.

Provides a standalone ``Reranker`` class and a ``RerankedRetriever``
that wraps any ``BaseRetriever`` to add a second-stage reranking step.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, DEFAULT_RERANK_TOP, DEFAULT_TOP_K
from data_models import Chunk, RetrievalResult
from retrieval.base_retriever import BaseRetriever


class Reranker:
    """Cross-encoder reranker (stateless scoring utility)."""

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        print(f"[reranker] Loading {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = DEFAULT_RERANK_TOP,
    ) -> tuple[list[Chunk], list[float]]:
        """Re-score *chunks* with the cross-encoder and return best *top_k*.

        Parameters
        ----------
        query : str
            Original query text.
        chunks : list[Chunk]
            Candidate chunks from a first-stage retriever.
        top_k : int
            How many to keep after reranking.

        Returns
        -------
        (reranked_chunks, scores)
        """
        if not chunks:
            return [], []
        pairs = [(query, c.text) for c in chunks]
        ce_scores = self.model.predict(pairs)
        order = np.argsort(ce_scores)[::-1][:top_k]
        return (
            [chunks[i] for i in order],
            [float(ce_scores[i]) for i in order],
        )


class RerankedRetriever(BaseRetriever):
    """Wraps any ``BaseRetriever`` and applies cross-encoder reranking.

    The base retriever produces an initial candidate set (``initial_k``),
    then the cross-encoder keeps the best ``final_k`` results.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Reranker,
        initial_k: int = DEFAULT_TOP_K * 2,
        final_k: int = DEFAULT_RERANK_TOP,
    ) -> None:
        self.base = base_retriever
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k

    @property
    def name(self) -> str:
        return f"{self.base.name} + Reranker"

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        if top_k is None:
            top_k = self.final_k

        # First stage: broad retrieval
        initial = self.base.retrieve(query, top_k=self.initial_k)

        # Second stage: cross-encoder reranking
        chunks, scores = self.reranker.rerank(
            query, initial.ranked_chunks, top_k=top_k,
        )
        return RetrievalResult(query=query, ranked_chunks=chunks, scores=scores)
