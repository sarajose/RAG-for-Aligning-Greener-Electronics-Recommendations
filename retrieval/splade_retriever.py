"""
SPLADE sparse/neural retriever.

Implements a first working SPLADE baseline using token-level sparse
expansion from a masked language model. The class follows the same
BaseRetriever interface as BM25/dense/hybrid so evaluation code can
reuse it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import DEFAULT_TOP_K, SPLADE_MAX_LENGTH, SPLADE_MODEL
from data_models import Chunk, RetrievalResult
from embedding_indexing import load_indices
from retrieval.base_retriever import BaseRetriever


@dataclass
class _SparseVector:
    token_ids: np.ndarray
    weights: np.ndarray


def _dot_sparse(a: _SparseVector, b: _SparseVector) -> float:
    """Compute sparse dot product for sorted token-id vectors."""
    i, j = 0, 0
    score = 0.0
    a_ids, a_w = a.token_ids, a.weights
    b_ids, b_w = b.token_ids, b.weights

    while i < len(a_ids) and j < len(b_ids):
        ai = a_ids[i]
        bj = b_ids[j]
        if ai == bj:
            score += float(a_w[i] * b_w[j])
            i += 1
            j += 1
        elif ai < bj:
            i += 1
        else:
            j += 1
    return score


class SPLADERetriever(BaseRetriever):
    """Sparse neural retriever based on SPLADE term expansions."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        model_name: str = SPLADE_MODEL,
        max_length: int = SPLADE_MAX_LENGTH,
        device: str | None = None,
    ) -> None:
        self.chunks = chunks
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[splade] Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Pre-encode corpus chunks once so retrieval can reuse the vectors.
        self.doc_vectors = self._encode_texts([c.text for c in chunks], is_query=False)

    @property
    def name(self) -> str:
        return "SPLADE"

    @classmethod
    def from_disk(
        cls,
        model_key: str,
        *,
        model_name: str = SPLADE_MODEL,
        max_length: int = SPLADE_MAX_LENGTH,
        device: str | None = None,
    ) -> "SPLADERetriever":
        """Build SPLADE over chunks loaded from existing index artifacts."""
        _, _, chunks = load_indices(model_key)
        return cls(chunks, model_name=model_name, max_length=max_length, device=device)

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        model_name: str = SPLADE_MODEL,
        max_length: int = SPLADE_MAX_LENGTH,
        device: str | None = None,
    ) -> "SPLADERetriever":
        """Build SPLADE from in-memory chunks (used by MTEB path)."""
        return cls(chunks, model_name=model_name, max_length=max_length, device=device)

    def _encode_texts(self, texts: Iterable[str], *, is_query: bool) -> list[_SparseVector]:
        vectors: list[_SparseVector] = []
        for text in texts:
            vectors.append(self._encode_one(text, is_query=is_query))
        return vectors

    def _encode_one(self, text: str, *, is_query: bool) -> _SparseVector:
        # Keep different prefixes explicit for methodological parity with E5.
        prefixed = f"query: {text}" if is_query else f"passage: {text}"
        inputs = self.tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, seq_len, vocab]
            attn = inputs["attention_mask"].unsqueeze(-1)
            # SPLADE pooling: max over sequence of log(1 + relu(logits)).
            sparse = torch.log1p(torch.relu(logits)) * attn
            pooled = torch.max(sparse, dim=1).values.squeeze(0).cpu().numpy()

        nz = np.nonzero(pooled > 0)[0]
        if len(nz) == 0:
            return _SparseVector(token_ids=np.array([], dtype=np.int32), weights=np.array([], dtype=np.float32))

        token_ids = nz.astype(np.int32)
        weights = pooled[nz].astype(np.float32)
        order = np.argsort(token_ids)
        return _SparseVector(token_ids=token_ids[order], weights=weights[order])

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        q_vec = self._encode_one(query, is_query=True)
        scores = np.array([_dot_sparse(q_vec, d_vec) for d_vec in self.doc_vectors], dtype=np.float32)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[i] for i in top_idx],
            scores=[float(scores[i]) for i in top_idx],
        )
