"""
BGE-M3 ColBERT multi-vector retriever.

Uses FlagEmbedding's BGEM3FlagModel in ColBERT mode (late-interaction
MaxSim scoring) for token-level relevance matching.  This is especially
effective for EU legal text where article references and specific legal
terms drive relevance more than whole-passage semantics.

Interface mirrors SPLADE/Dense retrievers so evaluation code reuses it
directly via --include-colbert.

Requirements:
    pip install FlagEmbedding

Memory note:
    At 600-token chunks (~200 actual BPE tokens), each chunk's ColBERT
    matrix is ~(200, 1024) float16 ≈ 400 KB.  For 10 000 chunks this
    is ~4 GB; kept on CPU and moved to GPU per query batch.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_TOP_K, INDEX_DIR
from data_models import Chunk, RetrievalResult
from embedding_indexing import load_indices
from retrieval.base_retriever import BaseRetriever

_BGE_M3_MODEL_ID = "BAAI/bge-m3"


def _maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """MaxSim score between one query and one document.

    Parameters
    ----------
    query_vecs : np.ndarray, shape (q_tokens, dim)
    doc_vecs   : np.ndarray, shape (d_tokens, dim)

    Returns
    -------
    float — ColBERT MaxSim score
    """
    # (q_tokens, d_tokens)
    sim = query_vecs @ doc_vecs.T
    # max over document tokens for each query token, then sum
    return float(sim.max(axis=1).sum())


class ColBERTRetriever(BaseRetriever):
    """BGE-M3 late-interaction ColBERT retriever.

    Chunks are pre-encoded once (token-level embeddings stored in
    ``self._doc_vecs``).  At query time MaxSim is computed against all
    stored chunk matrices and the top-k are returned.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        doc_vecs: list[np.ndarray],
        model_id: str = _BGE_M3_MODEL_ID,
    ) -> None:
        self.chunks = chunks
        self._doc_vecs = doc_vecs   # list of (n_tokens, 1024) float16 arrays
        self._model_id = model_id

    @property
    def name(self) -> str:
        return "BGE-M3 ColBERT (multi-vector)"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        model_id: str = _BGE_M3_MODEL_ID,
        batch_size: int = 16,
        cache_path: Path | None = None,
    ) -> "ColBERTRetriever":
        """Encode chunks with ColBERT and (optionally) cache to disk."""
        if cache_path is not None and cache_path.exists():
            print(f"[ColBERT] Loading cached token vecs from {cache_path}")
            with open(cache_path, "rb") as f:
                doc_vecs = pickle.load(f)
            return cls(chunks, doc_vecs, model_id)

        print(f"[ColBERT] Loading {model_id} in ColBERT mode …")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for ColBERT retrieval. "
                "Install with: pip install FlagEmbedding"
            ) from exc

        model = BGEM3FlagModel(model_id, use_fp16=True)
        texts = [c.text for c in chunks]
        doc_vecs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            out = model.encode(
                batch,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            for vec in out["colbert_vecs"]:
                doc_vecs.append(vec.astype(np.float16))
            if start % (batch_size * 10) == 0:
                print(f"[ColBERT]   encoded {start + len(batch)}/{len(texts)} chunks")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(doc_vecs, f)
            print(f"[ColBERT] Token vecs cached to {cache_path}")

        return cls(chunks, doc_vecs, model_id)

    @classmethod
    def from_disk(
        cls,
        model_key: str = "bge-m3",
        model_id: str = _BGE_M3_MODEL_ID,
        batch_size: int = 16,
    ) -> "ColBERTRetriever":
        """Load chunk list from a pre-built index pkl and encode with ColBERT."""
        _, _, chunks = load_indices(model_key)
        cache_path = INDEX_DIR / f"{model_key}_colbert_vecs.pkl"
        return cls.from_chunks(chunks, model_id, batch_size, cache_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for ColBERT retrieval. "
                "Install with: pip install FlagEmbedding"
            ) from exc

        model = BGEM3FlagModel(self._model_id, use_fp16=True)
        q_out = model.encode(
            [query],
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        q_vecs = q_out["colbert_vecs"][0].astype(np.float32)

        scores = np.array([
            _maxsim(q_vecs, d.astype(np.float32)) for d in self._doc_vecs
        ])
        top_idx = np.argsort(scores)[::-1][:top_k]

        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[i] for i in top_idx],
            scores=[float(scores[i]) for i in top_idx],
        )