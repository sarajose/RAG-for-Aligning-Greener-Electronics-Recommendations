"""
Dense semantic retriever using FAISS vector search.

Uses sentence-transformer embeddings indexed via FAISS HNSW or flat
inner-product search.  This is the dense-only baseline.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import DEFAULT_MODEL_KEY, DEFAULT_TOP_K
from data_models import Chunk, RetrievalResult
from embedding_indexing import get_embed_model, embed_texts, load_indices
from retrieval.base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense FAISS retriever (semantic similarity)."""

    def __init__(
        self,
        faiss_index: faiss.Index,
        chunks: list[Chunk],
        embed_model: SentenceTransformer,
    ) -> None:
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.embed_model = embed_model

    @property
    def name(self) -> str:
        return "Dense (FAISS)"

    @classmethod
    def from_disk(cls, model_key: str = DEFAULT_MODEL_KEY) -> "DenseRetriever":
        """Load pre-built FAISS index, chunks, and embedding model."""
        fi, _, chunks = load_indices(model_key)
        embed_model = get_embed_model(model_key)
        return cls(fi, chunks, embed_model)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        q_emb = embed_texts([query], self.embed_model, show_progress=False)
        scores, indices = self.faiss_index.search(
            q_emb.astype(np.float32), top_k,
        )
        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[i] for i in indices[0]],
            scores=[float(s) for s in scores[0]],
        )
