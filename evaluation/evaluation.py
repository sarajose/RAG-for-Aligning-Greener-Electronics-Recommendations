"""Core evaluation utilities for the simplified thesis pipeline.

This module intentionally keeps only the essential functionality needed by:
- prompting/retrieval workflows
- unified robust retrieval evaluation
- notebook visualizations fed by unified-eval outputs
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import DEFAULT_RERANK_TOP, DEFAULT_TOP_K, EVAL_K_VALUES, GOLD_STANDARD_CSV
from data_models import GoldStandardEntry, RetrievalMetrics
from evaluation.metrics import (
    average_precision,
    compute_retrieval_metrics,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)


def _read_text_with_fallback_encodings(path: Path) -> str:
    """Read text with a small ordered set of encodings for Windows CSV compatibility."""
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error: UnicodeDecodeError | None = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    # Should be unreachable because latin-1 can decode any byte sequence.
    if last_error is not None:
        raise last_error
    return path.read_text(encoding="utf-8")


def _pick(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return default


def load_gold_standard(csv_path: Path = GOLD_STANDARD_CSV) -> list[GoldStandardEntry]:
    """Load gold-standard entries from CSV with tolerant column handling."""
    entries: list[GoldStandardEntry] = []
    text = _read_text_with_fallback_encodings(csv_path)
    for row in csv.DictReader(text.splitlines()):
        entries.append(
            GoldStandardEntry(
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
            )
        )
    logger.info("Loaded %d gold-standard rows from %s", len(entries), csv_path)
    return entries


def group_gold_by_query(entries: list[GoldStandardEntry]) -> dict[str, set[str]]:
    """Group gold labels into {query: {canonical document names}}."""
    from config import normalise_doc_name

    grouped: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        query = (e.recommendation_text or "").strip()
        if not query:
            continue
        grouped[query].add(normalise_doc_name(e.doc_short_name))
    return dict(grouped)


def evaluate_retrieval(
    retriever,
    gold_path: Path = GOLD_STANDARD_CSV,
    k_values: Optional[list[int]] = None,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
) -> dict[int, RetrievalMetrics]:
    """Evaluate document-level retrieval against the gold standard."""
    from config import normalise_doc_name

    if k_values is None:
        k_values = EVAL_K_VALUES

    entries = load_gold_standard(gold_path)
    query_to_docs = group_gold_by_query(entries)
    queries = sorted(query_to_docs.keys())

    max_k = max(k_values)
    n_retrieve = max(max_k * 3, top_k_retrieve, 30)
    n_rerank = max(max_k, rerank_top)

    all_retrieved_docs: list[list[str]] = []
    all_relevant_docs: list[set[str]] = []

    for query in queries:
        try:
            result = retriever.retrieve(query, top_k=n_retrieve, rerank_top=n_rerank)
        except TypeError:
            result = retriever.retrieve(query, top_k=n_rerank)

        seen: set[str] = set()
        doc_ranking: list[str] = []
        for chunk in result.ranked_chunks:
            canon = normalise_doc_name(chunk.document)
            if canon not in seen:
                seen.add(canon)
                doc_ranking.append(canon)

        all_retrieved_docs.append(doc_ranking)
        all_relevant_docs.append(query_to_docs[query])

    return {
        k: compute_retrieval_metrics(all_retrieved_docs, all_relevant_docs, k)
        for k in sorted(set(k_values))
    }


def per_query_retrieval_scores(
    retriever,
    gold_path: Path = GOLD_STANDARD_CSV,
    k: int = 5,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
    level: str = "document",
) -> dict[str, list[float]]:
    """Return per-query scores used for CI and significance tests."""
    from config import normalise_doc_name

    entries = load_gold_standard(gold_path)
    query_to_docs = group_gold_by_query(entries)
    queries = sorted(query_to_docs.keys())

    n_retrieve = max(k * 3, top_k_retrieve, 30)
    n_rerank = max(k, rerank_top)

    out: dict[str, list[float]] = {
        m: [] for m in ("hit", "recall", "precision", "mrr", "ap", "ndcg", "rank")
    }

    for query in queries:
        try:
            result = retriever.retrieve(query, top_k=n_retrieve, rerank_top=n_rerank)
        except TypeError:
            result = retriever.retrieve(query, top_k=n_rerank)

        relevant = query_to_docs[query]

        if level == "document":
            seen: set[str] = set()
            ids: list[str] = []
            for c in result.ranked_chunks:
                canon = normalise_doc_name(c.document)
                if canon not in seen:
                    seen.add(canon)
                    ids.append(canon)
            rel_set = relevant
        else:
            ids = [c.id for c in result.ranked_chunks]
            rel_set = {
                c.id
                for c in result.ranked_chunks
                if normalise_doc_name(c.document) in relevant
            }

        out["hit"].append(float(hit_at_k(ids[:k], rel_set)))
        out["recall"].append(float(recall_at_k(ids[:k], rel_set)))
        out["precision"].append(float(precision_at_k(ids, rel_set, k)))
        out["mrr"].append(float(reciprocal_rank(ids[:k], rel_set)))
        out["ap"].append(float(average_precision(ids[:k], rel_set)))
        out["ndcg"].append(float(ndcg_at_k(ids, rel_set, k)))
        out["rank"].append(float(rank_of_first_relevant(ids[:k], rel_set)))

    return out


def load_whitepaper_recommendations(csv_path: Path) -> list[dict[str, str]]:
    """Load whitepaper recommendations from a semicolon-delimited CSV."""
    rows: list[dict[str, str]] = []
    text = _read_text_with_fallback_encodings(csv_path)
    for row in csv.DictReader(text.splitlines(), delimiter=";"):
        rows.append(
            {
                "section": (row.get("section") or "").strip(),
                "subsection": (row.get("subsection") or "").strip(),
                "title": (row.get("title") or "").strip(),
                "recommendation": (row.get("recommendation") or "").strip(),
            }
        )
    return rows


def format_retrieval_report(
    metrics_by_k: dict[int, RetrievalMetrics],
    title: str = "Retrieval Evaluation",
) -> str:
    """Format retrieval metrics in a compact table."""
    if not metrics_by_k:
        return f"{title}: no metrics"

    m0 = next(iter(metrics_by_k.values()))
    lines = [
        "",
        "=" * 82,
        f"  {title}  (n = {m0.num_queries} queries)",
        "=" * 82,
        f"{'k':>4} | {'Hit@k':>7} | {'Recall':>7} | {'Prec':>7} | {'MRR':>7} | {'MAP':>7} | {'NDCG':>7} | {'MR':>7}",
        "-" * 82,
    ]

    for k, m in sorted(metrics_by_k.items()):
        mr_str = f"{m.mean_rank:>7.1f}" if m.mean_rank != float("inf") else "    inf"
        lines.append(
            f"{k:>4} | {m.hit_rate:>7.3f} | {m.recall:>7.3f} | {m.precision:>7.3f} | "
            f"{m.mrr:>7.3f} | {m.map_score:>7.3f} | {m.ndcg:>7.3f} | {mr_str}"
        )

    lines.append("=" * 82)
    return "\n".join(lines)
