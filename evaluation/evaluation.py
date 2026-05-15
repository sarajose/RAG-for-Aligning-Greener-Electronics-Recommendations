"""Evaluation utilities: metrics, gold standard loading, and evaluation helpers."""

from __future__ import annotations

import csv
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_RERANK_TOP,
    DEFAULT_TOP_K,
    EVAL_K_VALUES,
    GOLD_STANDARD_CSV,
    INDEX_DIR,
)
from data_models import GoldStandardEntry, RetrievalMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _read_text_with_fallback_encodings(path: Path) -> str:
    """Read a text file trying utf-8, utf-8-sig, cp1252, then latin-1."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def _detect_delimiter(sample: str) -> str:
    """Sniff CSV delimiter from the first few lines; default to comma."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except csv.Error:
        return ","


def _pick(row: dict[str, Any], *keys: str, default: str = "") -> str:
    """Return the first non-None value found for any of the given keys."""
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return default


def _normalize_ws(text: str) -> str:
    """Collapse whitespace and strip a string."""
    return " ".join((text or "").strip().split())


# ---------------------------------------------------------------------------
# Per-query retrieval metrics
# ---------------------------------------------------------------------------

def hit_at_k(retrieved: list[str], relevant: set[str]) -> int:
    """1 if any relevant document appears in the retrieved list, else 0."""
    return int(bool(relevant & set(retrieved)))


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of relevant documents found in retrieved."""
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def rank_of_first_relevant(retrieved: list[str], relevant: set[str]) -> float:
    """1-based rank of the first relevant document; inf if none found."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return float(rank)
    return float("inf")


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Mean precision at each rank position where a relevant document is retrieved."""
    if not relevant:
        return 0.0
    hits, sum_prec = 0, 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain at cutoff k."""
    def _dcg(rels: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    rels = [1 if d in relevant else 0 for d in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregated retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
    k: int,
    chunk_hit_rate: float = 0.0,
) -> RetrievalMetrics:
    """Aggregate per-query retrieval metrics over a full query set at cutoff k."""
    n = len(all_retrieved)
    if n != len(all_relevant):
        raise ValueError(f"Length mismatch: {n} retrieved vs {len(all_relevant)} relevant")

    hits    = [hit_at_k(r[:k], rel)         for r, rel in zip(all_retrieved, all_relevant)]
    recalls = [recall_at_k(r[:k], rel)       for r, rel in zip(all_retrieved, all_relevant)]
    precs   = [precision_at_k(r, rel, k)     for r, rel in zip(all_retrieved, all_relevant)]
    mrrs    = [reciprocal_rank(r[:k], rel)   for r, rel in zip(all_retrieved, all_relevant)]
    aps     = [average_precision(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    ndcgs   = [ndcg_at_k(r, rel, k)         for r, rel in zip(all_retrieved, all_relevant)]
    ranks   = [rank_of_first_relevant(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]

    found_ranks = [r for r in ranks if r != float("inf")]
    return RetrievalMetrics(
        k=k,
        hit_rate=float(np.mean(hits)),
        recall=float(np.mean(recalls)),
        precision=float(np.mean(precs)),
        mrr=float(np.mean(mrrs)),
        map_score=float(np.mean(aps)),
        ndcg=float(np.mean(ndcgs)),
        num_queries=n,
        mean_rank=float(np.mean(found_ranks)) if found_ranks else float("inf"),
        chunk_hit_rate=chunk_hit_rate,
    )


# ---------------------------------------------------------------------------
# Shared helpers (used by retrieval_eval.py and mteb_eval.py)
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Current timestamp string for logging."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(message: str) -> None:
    """Print a timestamped progress line (used by MTEB eval)."""
    print(f"[{_ts()}] [mteb] {message}", flush=True)


def _safe_retrieve(retriever: Any, query: str, top_k: int):
    """Call retriever.retrieve with top_k, falling back to positional arg if needed."""
    try:
        return retriever.retrieve(query, top_k=top_k)
    except TypeError:
        return retriever.retrieve(query)


def _metrics_to_rows(
    metrics_by_k: dict[int, Any],
    *,
    dataset: str,
    level: str,
    model_key: str,
    method: str,
) -> list[dict]:
    """Convert a {k: RetrievalMetrics} dict into flat dicts ready for a DataFrame."""
    rows: list[dict] = []
    for k, m in sorted(metrics_by_k.items()):
        rows.append({
            "dataset": dataset, "level": level, "model_key": model_key, "method": method,
            "k": k, "hit_rate": m.hit_rate, "recall": m.recall, "precision": m.precision,
            "mrr": m.mrr, "map": m.map_score, "ndcg": m.ndcg,
            "mean_rank": m.mean_rank, "chunk_hit_rate": m.chunk_hit_rate, "num_queries": m.num_queries,
        })
    return rows


def _indices_exist(model_key: str) -> bool:
    """Return True if all three index artifacts (FAISS, BM25, chunks) exist for model_key."""
    prefix = INDEX_DIR / model_key
    return (
        Path(str(prefix) + "_faiss.index").exists()
        and Path(str(prefix) + "_bm25.pkl").exists()
        and Path(str(prefix) + "_chunks.pkl").exists()
    )


# ---------------------------------------------------------------------------
# Gold standard loading
# ---------------------------------------------------------------------------

def load_gold_standard(csv_path: Path = GOLD_STANDARD_CSV) -> list[GoldStandardEntry]:
    """Load gold-standard annotation rows from CSV with tolerant encoding/delimiter handling."""
    entries: list[GoldStandardEntry] = []
    text = _read_text_with_fallback_encodings(csv_path)
    delimiter = _detect_delimiter("\n".join(text.splitlines()[:20]))
    for row in csv.DictReader(text.splitlines(), delimiter=delimiter):
        entries.append(GoldStandardEntry(
            paper=_pick(row, "Paper"),
            source_page=_pick(row, "source_page"),
            source_line=_pick(row, "source_line"),
            recommendation_text=_pick(row, "recommendation_text", "recommendation"),
            source_snippet_original=_pick(row, "source_snippet_original"),
            recommendation_or_statement=_pick(row, "recommendation_or_statement"),
            doc_short_name=_pick(row, "doc_short_name", "legal_doc_reference", "document"),
            doc_type=_pick(row, "doc_type"),
            doc_ref_num=_pick(row, "doc_ref_num"),
            doc_reference_raw_excerpt=_pick(row, "doc_reference_raw_excerpt"),
            evidence_span=_pick(row, "evidence_span"),
            reference_basis=_pick(row, "reference_basis"),
            needs_review=_pick(row, "needs_review"),
            context_excerpt=_pick(row, "context_excerpt"),
            alignment_label=_pick(row, "alignment_label") or None,
        ))
    logger.info("Loaded %d gold-standard rows from %s", len(entries), csv_path)
    return entries


def group_gold_query_instances(entries: list[GoldStandardEntry]) -> list[dict[str, Any]]:
    """Group gold entries by (recommendation_text, source_snippet) and return query instances."""
    from config import normalise_doc_name

    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for e in entries:
        query = _normalize_ws(e.recommendation_text)
        if not query:
            continue
        snippet = _normalize_ws(e.source_snippet_original)
        grouped[(query, snippet)].add(normalise_doc_name(e.doc_short_name))
    return [
        {"query": query, "source_snippet_original": snippet, "relevant_docs": set(docs)}
        for (query, snippet), docs in sorted(grouped.items(), key=lambda x: x[0])
    ]


# ---------------------------------------------------------------------------
# Core document-level retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retriever,
    gold_path: Path = GOLD_STANDARD_CSV,
    k_values: Optional[list[int]] = None,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
) -> dict[int, RetrievalMetrics]:
    """Evaluate document-level retrieval against gold standard at multiple k cutoffs."""
    from config import normalise_doc_name

    if k_values is None:
        k_values = EVAL_K_VALUES

    entries = load_gold_standard(gold_path)
    query_instances = group_gold_query_instances(entries)
    max_k = max(k_values)
    n_retrieve = max(max_k * 3, top_k_retrieve, 30)
    n_rerank = max(max_k, rerank_top)

    all_retrieved_docs: list[list[str]] = []
    all_relevant_docs: list[set[str]] = []
    chunk_hits_top1: list[int] = []

    for item in query_instances:
        query = item["query"]
        try:
            result = retriever.retrieve(query, top_k=n_retrieve, rerank_top=n_rerank)
        except TypeError:
            result = retriever.retrieve(query, top_k=n_rerank)

        relevant_docs = item["relevant_docs"]
        top1_doc = normalise_doc_name(result.ranked_chunks[0].document) if result.ranked_chunks else ""
        chunk_hits_top1.append(1 if top1_doc in relevant_docs else 0)

        seen: set[str] = set()
        doc_ranking: list[str] = []
        for chunk in result.ranked_chunks:
            canon = normalise_doc_name(chunk.document)
            if canon not in seen:
                seen.add(canon)
                doc_ranking.append(canon)

        all_retrieved_docs.append(doc_ranking)
        all_relevant_docs.append(relevant_docs)

    chunk_hit_rate = float(sum(chunk_hits_top1) / len(chunk_hits_top1)) if chunk_hits_top1 else 0.0
    return {
        k: compute_retrieval_metrics(all_retrieved_docs, all_relevant_docs, k, chunk_hit_rate=chunk_hit_rate)
        for k in sorted(set(k_values))
    }
