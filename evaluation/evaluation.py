"""Evaluation utilities: metrics, statistical tests, gold standard loading, exports, and ablation.

This module consolidates all evaluation logic so callers only need to import from here:
- Retrieval and classification metrics (hit@k, NDCG, MRR, F1, etc.)
- Bootstrap CI and paired permutation tests
- Gold standard loading and query grouping
- Per-query scoring and document-level evaluation
- CSV exports for gold/whitepaper retrieved chunks
- Ablation table construction and significance annotation
- Shared utility helpers used by retrieval_eval and mteb_eval
"""

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
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import (
    DEFAULT_RERANK_TOP,
    DEFAULT_TOP_K,
    EVAL_K_VALUES,
    GOLD_STANDARD_CSV,
    INDEX_DIR,
)
from data_models import ClassificationMetrics, GoldStandardEntry, RetrievalMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private CSV helpers
# ---------------------------------------------------------------------------

def _read_text_with_fallback_encodings(path: Path) -> str:
    """Read text trying multiple encodings for Windows CSV compatibility."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except csv.Error:
        return ","


def _pick(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None:
            return str(row[k])
    return default


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


# ---------------------------------------------------------------------------
# Retrieval metrics (per-query)
# ---------------------------------------------------------------------------

def hit_at_k(retrieved: list[str], relevant: set[str]) -> int:
    """1 if any relevant item appears in retrieved, else 0."""
    return int(bool(relevant & set(retrieved)))


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def rank_of_first_relevant(retrieved: list[str], relevant: set[str]) -> float:
    """1-based rank of the first relevant item; inf if none found."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return float(rank)
    return float("inf")


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def _dcg(rels: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    rels = [1 if d in relevant else 0 for d in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Retrieval metrics (aggregated)
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
    k: int,
    chunk_hit_rate: float = 0.0,
) -> RetrievalMetrics:
    """Aggregate retrieval metrics over a query set at cutoff *k*."""
    n = len(all_retrieved)
    if n != len(all_relevant):
        raise ValueError(f"Length mismatch: {n} retrieved vs {len(all_relevant)} relevant")

    hits = [hit_at_k(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    recalls = [recall_at_k(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    precs = [precision_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    mrrs = [reciprocal_rank(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    aps = [average_precision(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    ndcgs = [ndcg_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    ranks = [rank_of_first_relevant(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]

    found_ranks = [r for r in ranks if r != float("inf")]
    mean_r = float(np.mean(found_ranks)) if found_ranks else float("inf")

    return RetrievalMetrics(
        k=k,
        hit_rate=float(np.mean(hits)),
        recall=float(np.mean(recalls)),
        precision=float(np.mean(precs)),
        mrr=float(np.mean(mrrs)),
        map_score=float(np.mean(aps)),
        ndcg=float(np.mean(ndcgs)),
        num_queries=n,
        mean_rank=mean_r,
        chunk_hit_rate=chunk_hit_rate,
    )


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> ClassificationMetrics:
    """Accuracy, macro/weighted F1, Cohen's kappa, and per-class breakdown."""
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = {
        lbl: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for lbl, p, r, f in zip(labels, prec_arr, rec_arr, f1_arr)
    }
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return ClassificationMetrics(
        accuracy=float(acc),
        macro_f1=float(macro),
        weighted_f1=float(weighted),
        cohens_kappa=float(kappa),
        per_class=per_class,
        confusion_matrix=cm,
        labels=labels,
        num_samples=len(y_true),
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of *scores*. Returns (mean, ci_lower, ci_upper)."""
    rng = np.random.RandomState(rng_seed)
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    means = np.array([arr[rng.randint(0, n, size=n)].mean() for _ in range(n_bootstrap)])
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def paired_permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10_000,
    rng_seed: int = 42,
) -> float:
    """Two-sided paired permutation test. Returns p-value."""
    rng = np.random.RandomState(rng_seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert len(a) == len(b), "Score lists must have the same length"
    diff = a - b
    observed = np.abs(diff.mean())
    count = sum(
        1 for _ in range(n_permutations)
        if np.abs((diff * rng.choice([-1, 1], size=len(diff))).mean()) >= observed
    )
    return count / n_permutations


def _paired_effect_size_dz(scores_a: list[float], scores_b: list[float]) -> float:
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    diff = a - b
    if len(diff) < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    return float(np.mean(diff) / sd) if sd != 0.0 else 0.0


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    sorted_p = [float(p_values[i]) for i in order]
    adjusted_sorted: list[float] = []
    prev = 0.0
    for i, p in enumerate(sorted_p):
        adj = max(min(1.0, (m - i) * p), prev)
        adjusted_sorted.append(adj)
        prev = adj
    adjusted: list[float] = [0.0] * m
    for sorted_idx, original_idx in enumerate(order):
        adjusted[int(original_idx)] = adjusted_sorted[sorted_idx]
    return adjusted


def _effect_size_label(dz: float) -> str:
    adz = abs(dz)
    if adz < 0.2:
        return "negligible"
    if adz < 0.5:
        return "small"
    if adz < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Shared utility helpers (used by retrieval_eval and mteb_eval)
# ---------------------------------------------------------------------------

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(message: str) -> None:
    print(f"[{_ts()}] [mteb] {message}", flush=True)


def _safe_retrieve(retriever: Any, query: str, top_k: int):
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
    prefix = INDEX_DIR / model_key
    return (
        Path(str(prefix) + "_faiss.index").exists()
        and Path(str(prefix) + "_bm25.pkl").exists()
        and Path(str(prefix) + "_chunks.pkl").exists()
    )


def _build_metrics_summary_tables(
    metrics_df: pd.DataFrame,
    k_for_summary: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = (
        metrics_df[metrics_df["k"] == k_for_summary][
            ["dataset", "level", "model_key", "method", "k", "hit_rate", "mrr", "ndcg", "num_queries"]
        ]
        .copy()
        .sort_values(["dataset", "level", "method", "ndcg"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    comp_rows: list[dict[str, Any]] = []
    for (dataset, level, method), group in summary_df.groupby(["dataset", "level", "method"], dropna=False):
        for metric in ("ndcg", "mrr", "hit_rate"):
            ranked = group.sort_values(metric, ascending=False).reset_index(drop=True)
            best = ranked.iloc[0]
            if len(ranked) >= 2:
                second_model = str(ranked.iloc[1]["model_key"])
                second_value = float(ranked.iloc[1][metric])
                gap = float(best[metric] - ranked.iloc[1][metric])
            else:
                second_model, second_value, gap = "", float("nan"), float("nan")
            comp_rows.append({
                "dataset": dataset, "level": level, "method": method, "k": int(k_for_summary),
                "metric": metric, "best_model": str(best["model_key"]), "best_value": float(best[metric]),
                "second_model": second_model, "second_value": second_value, "gap_to_second": gap,
                "num_models_compared": int(len(ranked)),
            })
    comparison_df = pd.DataFrame(comp_rows, columns=[
        "dataset", "level", "method", "k", "metric",
        "best_model", "best_value", "second_model", "second_value", "gap_to_second", "num_models_compared",
    ])
    return summary_df, comparison_df


def _validate_ranking_consistency(
    metrics_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    k_for_ranking: int = 10,
) -> None:
    sort_cols = ["dataset", "level", "ndcg", "model_key", "method"]
    sort_asc = [True, True, False, True, True]
    expected = (
        metrics_df[metrics_df["k"] == k_for_ranking]
        .sort_values(sort_cols, ascending=sort_asc, kind="mergesort")
        .reset_index(drop=True)
    )
    actual = ranking_df.copy().sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    if expected.shape != ranking_df.shape:
        raise RuntimeError(f"Ranking export shape mismatch: expected={expected.shape}, actual={ranking_df.shape}")
    if list(expected.columns) != list(actual.columns):
        raise RuntimeError(f"Ranking export columns mismatch.")
    try:
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        raise RuntimeError(f"Ranking export does not match evaluated metrics. Details: {exc}") from exc


# ---------------------------------------------------------------------------
# Gold standard loading
# ---------------------------------------------------------------------------

def load_gold_standard(csv_path: Path = GOLD_STANDARD_CSV) -> list[GoldStandardEntry]:
    """Load gold-standard entries from CSV with tolerant column/encoding handling."""
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
    """Group gold labels by (recommendation_text, source_snippet_original) instance."""
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


def group_gold_by_query(entries: list[GoldStandardEntry]) -> dict[str, set[str]]:
    """Collapse query instances by recommendation text only (legacy helper)."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in group_gold_query_instances(entries):
        grouped[item["query"]].update(item["relevant_docs"])
    return dict(grouped)


# ---------------------------------------------------------------------------
# Core retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    retriever,
    gold_path: Path = GOLD_STANDARD_CSV,
    k_values: Optional[list[int]] = None,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
) -> dict[int, RetrievalMetrics]:
    """Evaluate document-level retrieval against the gold standard at multiple k cutoffs."""
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


def per_query_retrieval_scores(
    retriever,
    gold_path: Path = GOLD_STANDARD_CSV,
    k: int = 5,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
    level: str = "document",
) -> dict[str, list[float]]:
    """Return per-query scores for CI and significance tests."""
    from config import normalise_doc_name

    entries = load_gold_standard(gold_path)
    query_instances = group_gold_query_instances(entries)
    n_retrieve = max(k * 3, top_k_retrieve, 30)
    n_rerank = max(k, rerank_top)
    out: dict[str, list[float]] = {m: [] for m in ("hit", "recall", "precision", "mrr", "ap", "ndcg", "rank")}

    for item in query_instances:
        query = item["query"]
        try:
            result = retriever.retrieve(query, top_k=n_retrieve, rerank_top=n_rerank)
        except TypeError:
            result = retriever.retrieve(query, top_k=n_rerank)

        relevant = item["relevant_docs"]
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
            rel_set = {c.id for c in result.ranked_chunks if normalise_doc_name(c.document) in relevant}

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
        rows.append({
            "section": (row.get("section") or "").strip(),
            "subsection": (row.get("subsection") or "").strip(),
            "title": (row.get("title") or "").strip(),
            "recommendation": (row.get("recommendation") or "").strip(),
        })
    return rows


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def export_gold_retrieved_chunks(
    *,
    retriever: Any,
    model_key: str,
    method: str,
    gold_csv: Path,
    out_csv: Path,
    top_k: int,
) -> None:
    """Export top-k retrieved chunks for every gold standard query to CSV."""
    from config import normalise_doc_name

    query_instances = group_gold_query_instances(load_gold_standard(gold_csv))
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(query_instances, start=1):
        query = item["query"]
        result = _safe_retrieve(retriever, query, top_k=top_k)
        if i % 50 == 0 or i == len(query_instances):
            print(f"[gold] {i}/{len(query_instances)} queries processed for {model_key}|{method}")
        for rank, (chunk, score) in enumerate(zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1):
            rows.append({
                "dataset": "gold_standard", "model_key": model_key, "method": method,
                "query_instance_id": i, "query": query,
                "source_snippet_original": item.get("source_snippet_original", ""),
                "rank": rank, "score": float(score), "chunk_id": chunk.id,
                "document": chunk.document, "document_canonical": normalise_doc_name(chunk.document),
                "article": chunk.article, "paragraph": chunk.paragraph, "text": chunk.text,
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def export_whitepaper_retrieved_chunks(
    *,
    retriever: Any,
    model_key: str,
    method: str,
    whitepaper_csv: Path,
    out_csv: Path,
    top_k: int,
) -> None:
    """Export top-k retrieved chunks for every whitepaper recommendation query to CSV."""
    from config import normalise_doc_name

    wp_rows = load_whitepaper_recommendations(whitepaper_csv)
    rows: list[dict[str, Any]] = []
    for i, wp in enumerate(wp_rows, start=1):
        query = (wp.get("recommendation", "") or "").strip()
        if not query:
            query = f"{wp.get('section', '')} {wp.get('subsection', '')} {wp.get('title', '')}".strip()
        result = _safe_retrieve(retriever, query, top_k=top_k)
        if i % 10 == 0 or i == len(wp_rows):
            print(f"[whitepaper] {i}/{len(wp_rows)} queries processed for {model_key}|{method}")
        for rank, (chunk, score) in enumerate(zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1):
            rows.append({
                "dataset": "whitepaper", "model_key": model_key, "method": method,
                "section": wp.get("section", ""), "subsection": wp.get("subsection", ""),
                "title": wp.get("title", ""), "query": query,
                "rank": rank, "score": float(score), "chunk_id": chunk.id,
                "document": chunk.document, "document_canonical": normalise_doc_name(chunk.document),
                "article": chunk.article, "paragraph": chunk.paragraph, "text": chunk.text,
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def _sig_star(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""


_METHOD_ORDER = ["bm25", "dense", "rrf", "bm25_rerank", "dense_rerank", "rrf_rerank"]


def collect_per_query_scores(
    gold_csv: Path,
    model_keys: Optional[list[str]] = None,
    k: int = 10,
    skip_reranker: bool = False,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Collect per-query retrieval scores for all ablation methods across models."""
    from config import DEFAULT_RERANK_TOP, DEFAULT_TOP_K, EMBEDDING_MODELS
    from embedding_indexing import get_embed_model, load_indices
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.hybrid_retriever import HybridRetriever as CompositeHybridRetriever
    from retrieval.reranker import Reranker, RerankedRetriever

    if model_keys is None:
        model_keys = list(EMBEDDING_MODELS.keys())

    reranker = None if skip_reranker else Reranker()
    entries = load_gold_standard(gold_csv)
    query_instances = group_gold_query_instances(entries)
    queries = [item["query"] for item in query_instances]
    top_k = max(k * 3, DEFAULT_TOP_K, 30)
    rerank_top = max(k, DEFAULT_RERANK_TOP)
    rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        faiss_index, bm25_index, chunks = load_indices(model_key)
        embed_model = get_embed_model(model_key)
        bm25_ret = BM25Retriever(bm25_index, chunks)
        dense_ret = DenseRetriever(faiss_index, chunks, embed_model)
        hybrid_ret = CompositeHybridRetriever(faiss_index, bm25_index, chunks, embed_model)
        retrievers: dict[str, Any] = {"bm25": bm25_ret, "dense": dense_ret, "rrf": hybrid_ret}
        if reranker is not None:
            retrievers["bm25_rerank"] = RerankedRetriever(bm25_ret, reranker, initial_k=top_k, final_k=rerank_top)
            retrievers["dense_rerank"] = RerankedRetriever(dense_ret, reranker, initial_k=top_k, final_k=rerank_top)
            retrievers["rrf_rerank"] = RerankedRetriever(hybrid_ret, reranker, initial_k=top_k, final_k=rerank_top)

        for method_name, retriever in retrievers.items():
            scores = per_query_retrieval_scores(retriever, gold_path=gold_csv, k=k, top_k_retrieve=top_k, rerank_top=rerank_top)
            for idx, query in enumerate(queries):
                rows.append({
                    "model_key": model_key, "method": method_name, "query_idx": idx, "query": query,
                    "hit": scores["hit"][idx], "recall": scores["recall"][idx],
                    "mrr": scores["mrr"][idx], "ap": scores["ap"][idx], "ndcg": scores["ndcg"][idx],
                })

    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def build_ablation_table(
    metrics_csv: Path,
    k: int = 10,
    metrics: Optional[list[str]] = None,
    dataset: str = "gold_standard",
    level: str = "document",
) -> pd.DataFrame:
    """Build method × (model, metric) pivot table from metrics_all.csv."""
    if metrics is None:
        metrics = ["ndcg", "mrr", "hit_rate"]
    df = pd.read_csv(metrics_csv)
    subset = df[(df["dataset"] == dataset) & (df["level"] == level) & (df["k"] == k)]
    if subset.empty:
        raise ValueError(f"No rows for dataset={dataset!r}, level={level!r}, k={k} in {metrics_csv}.")
    pivot = subset.pivot_table(index="method", columns="model_key", values=metrics, aggfunc="first")
    present = [m for m in _METHOD_ORDER if m in pivot.index]
    extras = [m for m in pivot.index if m not in _METHOD_ORDER]
    return pivot.loc[present + extras]


def add_significance_markers(
    ablation_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    primary_metric: str = "ndcg",
    baseline_method: str = "bm25",
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Annotate ablation table values with significance stars vs the BM25 baseline."""
    model_keys = per_query_df["model_key"].unique().tolist()
    sig: dict[tuple[str, str], str] = {}

    for model_key in model_keys:
        baseline_scores = per_query_df[
            (per_query_df["model_key"] == model_key) & (per_query_df["method"] == baseline_method)
        ][primary_metric].tolist()
        if not baseline_scores:
            continue
        for method in per_query_df[per_query_df["model_key"] == model_key]["method"].unique():
            if method == baseline_method:
                continue
            method_scores = per_query_df[
                (per_query_df["model_key"] == model_key) & (per_query_df["method"] == method)
            ][primary_metric].tolist()
            if len(method_scores) != len(baseline_scores):
                continue
            p = paired_permutation_test(method_scores, baseline_scores, n_permutations=n_permutations, rng_seed=rng_seed)
            sig[(model_key, method)] = _sig_star(float(p), alpha)

    result = ablation_df.copy().astype(object)
    for (row_method, col_model, metric), val in ablation_df.stack(level=[0, 1]).items():
        marker = sig.get((col_model, row_method), "")
        result.loc[row_method, (metric, col_model)] = f"{val:.3f}{marker}"
    return result


def format_ablation_report(
    ablation_df: pd.DataFrame,
    title: str = "Ablation Study - Retrieval Methods",
    random_metrics: Optional[dict[int, RetrievalMetrics]] = None,
    oracle_metrics: Optional[dict[int, RetrievalMetrics]] = None,
    k: int = 10,
) -> str:
    """Format ablation table as a plain-text report."""
    lines = ["", "=" * 100, f"  {title}", "=" * 100]

    def _extra_row(label: str, mdict: dict[int, RetrievalMetrics]) -> str:
        if k not in mdict:
            return ""
        m = mdict[k]
        return f"  {label:<28}  NDCG={m.ndcg:.3f}  MRR={m.mrr:.3f}  Hit={m.hit_rate:.3f}  (all models identical)"

    if random_metrics:
        lines.append(_extra_row("Random baseline", random_metrics))
    if oracle_metrics:
        lines.append(_extra_row("Oracle upper bound", oracle_metrics))
    if random_metrics or oracle_metrics:
        lines.append("-" * 100)

    lines += [ablation_df.to_string(), "", "  * p<0.05   ** p<0.01   *** p<0.001  (paired permutation test)", "=" * 100]
    return "\n".join(lines)
