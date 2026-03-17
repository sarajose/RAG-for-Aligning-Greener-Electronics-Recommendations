"""
Retrieval package — modular retriever implementations.

Each file handles one retrieval strategy:
- ``bm25_retriever``    — sparse lexical baseline (BM25Okapi)
- ``dense_retriever``   — dense semantic search (FAISS)
- ``hybrid_retriever``  — Reciprocal Rank Fusion (BM25 + FAISS)
- ``reranker``          — cross-encoder reranking wrapper
"""

from retrieval.base_retriever import BaseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker, RerankedRetriever
from retrieval.splade_retriever import SPLADERetriever

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "SPLADERetriever",
    "Reranker",
    "RerankedRetriever",
]
