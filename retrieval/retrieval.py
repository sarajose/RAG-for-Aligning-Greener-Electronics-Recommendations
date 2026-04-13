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

from collections import defaultdict
import re

import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

from config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP,
    RERANKER_MODEL,
    RRF_K,
    evidence_group_for_document,
)
import config as _config
from data_models import Chunk, RetrievalResult
from embedding_indexing import (
    get_embed_model,
    embed_texts,
    load_indices,
    tokenize,
)

DEFAULT_MAX_CHUNKS_PER_DOC = int(getattr(_config, "DEFAULT_MAX_CHUNKS_PER_DOC", 0))
DEFAULT_NEAR_DUP_SUPPRESSION = bool(getattr(_config, "DEFAULT_NEAR_DUP_SUPPRESSION", False))

# Low-level search primitives

def search_faiss(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index; return (scores, indices) arrays of length k."""
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    return scores[0], indices[0]


def search_bm25(
    bm25: BM25Okapi,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Search a BM25Okapi index; return (scores, indices) arrays of length k."""
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
        self.chunk_groups = [evidence_group_for_document(c.document) for c in chunks]

        self._group_resources: dict[str, dict[str, object]] = {}
        group_to_indices: dict[str, list[int]] = {
            "binding_law": [],
            "policy_or_recommendation_docs": [],
        }
        for idx, grp in enumerate(self.chunk_groups):
            group_to_indices.setdefault(grp, []).append(idx)

        for group_name, global_indices in group_to_indices.items():
            if not global_indices:
                continue
            group_texts = [self.chunks[i].text for i in global_indices]
            group_embeddings = embed_texts(
                group_texts,
                self.embed_model,
                show_progress=False,
                is_query=False,
            )
            group_faiss = faiss.IndexFlatIP(group_embeddings.shape[1])
            group_faiss.add(group_embeddings.astype(np.float32))
            group_bm25 = BM25Okapi([tokenize(text) for text in group_texts])
            self._group_resources[group_name] = {
                "global_indices": global_indices,
                "faiss": group_faiss,
                "bm25": group_bm25,
            }

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
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        max_chunks_per_doc: int = DEFAULT_MAX_CHUNKS_PER_DOC,
        near_dup_suppression: bool = DEFAULT_NEAR_DUP_SUPPRESSION,
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
        if retrieval_mode == "split_evidence_retrieval":
            return self._retrieve_split_evidence(
                query=query,
                top_k=top_k,
                rerank_top=rerank_top,
                max_chunks_per_doc=max_chunks_per_doc,
                near_dup_suppression=near_dup_suppression,
            )

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
            evidence_groups=[self.chunk_groups[i] for i in final_indices],
            retrieval_mode="flat_baseline",
        )

    @staticmethod
    def _chunk_text_fingerprint(text: str) -> str:
        """Return a coarse normalized fingerprint for near-duplicate checks."""
        norm = re.sub(r"\s+", " ", text.lower().strip())
        norm = re.sub(r"[^a-z0-9 ]", "", norm)
        return norm[:220]

    def _retrieve_group_candidates(
        self,
        query: str,
        q_emb: np.ndarray,
        group_name: str,
        top_k: int,
        rerank_top: int,
    ) -> list[tuple[int, float, str]]:
        """Retrieve candidates from one evidence group only."""
        resources = self._group_resources.get(group_name)
        if not resources:
            return []

        global_indices: list[int] = resources["global_indices"]  # type: ignore[index]
        group_faiss: faiss.Index = resources["faiss"]  # type: ignore[index]
        group_bm25: BM25Okapi = resources["bm25"]  # type: ignore[index]

        candidate_k = min(top_k * 2, len(global_indices))
        _, bm25_local_idx = search_bm25(group_bm25, query, k=candidate_k)
        _, dense_local_idx = search_faiss(group_faiss, q_emb, k=candidate_k)

        fused_local = reciprocal_rank_fusion(
            [bm25_local_idx.tolist(), dense_local_idx.tolist()]
        )
        hybrid_local = [local_idx for local_idx, _ in fused_local[:top_k]]
        hybrid_global = [global_indices[i] for i in hybrid_local]

        if not hybrid_global:
            return []

        if self.use_reranker:
            reranked = rerank(
                query,
                self.chunks,
                hybrid_global,
                top_k=min(rerank_top, len(hybrid_global)),
            )
            return [(idx, score, group_name) for idx, score in reranked]

        local_fused_scores = {local_idx: score for local_idx, score in fused_local}
        candidates: list[tuple[int, float, str]] = []
        for global_idx in hybrid_global[:rerank_top]:
            local_idx = global_indices.index(global_idx)
            candidates.append((global_idx, float(local_fused_scores.get(local_idx, 0.0)), group_name))
        return candidates

    def _select_with_constraints(
        self,
        candidates: list[tuple[int, float, str]],
        final_k: int,
        max_chunks_per_doc: int,
        near_dup_suppression: bool,
    ) -> list[tuple[int, float, str]]:
        """Select final chunks with group coverage and anti-dominance constraints."""
        selected: list[tuple[int, float, str]] = []
        selected_ids: set[int] = set()
        doc_counts: dict[str, int] = defaultdict(int)
        doc_fingerprints: dict[str, set[str]] = defaultdict(set)

        def can_select(idx: int) -> bool:
            chunk = self.chunks[idx]
            if max_chunks_per_doc > 0 and doc_counts[chunk.document] >= max_chunks_per_doc:
                return False
            if near_dup_suppression:
                fp = self._chunk_text_fingerprint(chunk.text)
                if fp in doc_fingerprints[chunk.document]:
                    return False
            return True

        def add_candidate(candidate: tuple[int, float, str]) -> None:
            idx, score, group = candidate
            selected.append((idx, score, group))
            selected_ids.add(idx)
            chunk = self.chunks[idx]
            doc_counts[chunk.document] += 1
            if near_dup_suppression:
                doc_fingerprints[chunk.document].add(self._chunk_text_fingerprint(chunk.text))

        # Coverage first: try to include both groups when possible.
        for group_name in ("binding_law", "policy_or_recommendation_docs"):
            for candidate in candidates:
                idx, _, grp = candidate
                if grp != group_name or idx in selected_ids:
                    continue
                if can_select(idx):
                    add_candidate(candidate)
                    break
            if len(selected) >= final_k:
                return selected[:final_k]

        # Fill remaining by score.
        for candidate in candidates:
            idx, _, _ = candidate
            if idx in selected_ids:
                continue
            if not can_select(idx):
                continue
            add_candidate(candidate)
            if len(selected) >= final_k:
                break

        return selected

    def _retrieve_split_evidence(
        self,
        query: str,
        top_k: int,
        rerank_top: int,
        max_chunks_per_doc: int,
        near_dup_suppression: bool,
    ) -> RetrievalResult:
        """Retrieve separately from binding law and policy docs, then merge."""
        q_emb = embed_texts(
            [query],
            self.embed_model,
            show_progress=False,
            is_query=True,
        )

        candidates: list[tuple[int, float, str]] = []
        for group_name in ("binding_law", "policy_or_recommendation_docs"):
            candidates.extend(
                self._retrieve_group_candidates(
                    query=query,
                    q_emb=q_emb,
                    group_name=group_name,
                    top_k=top_k,
                    rerank_top=rerank_top,
                )
            )

        if not candidates:
            return RetrievalResult(
                query=query,
                ranked_chunks=[],
                scores=[],
                evidence_groups=[],
                retrieval_mode="split_evidence_retrieval",
            )

        # Deduplicate same chunk index across groups and keep best score.
        best_by_idx: dict[int, tuple[float, str]] = {}
        for idx, score, group in candidates:
            prev = best_by_idx.get(idx)
            if prev is None or score > prev[0]:
                best_by_idx[idx] = (score, group)

        merged = sorted(
            [(idx, score, group) for idx, (score, group) in best_by_idx.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        selected = self._select_with_constraints(
            candidates=merged,
            final_k=rerank_top,
            max_chunks_per_doc=max_chunks_per_doc,
            near_dup_suppression=near_dup_suppression,
        )

        final_indices = [idx for idx, _, _ in selected]
        final_scores = [score for _, score, _ in selected]
        final_groups = [group for _, _, group in selected]

        return RetrievalResult(
            query=query,
            ranked_chunks=[self.chunks[i] for i in final_indices],
            scores=final_scores,
            evidence_groups=final_groups,
            retrieval_mode="split_evidence_retrieval",
        )
