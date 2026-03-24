"""Minimal publishable evaluation utilities.

This module is intentionally ablation-focused and keeps only the
functions used by evaluation.experiment_commands:
- collect_per_query_scores
- build_ablation_table
- add_significance_markers
- format_ablation_report
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_models import RetrievalMetrics
from evaluation.metrics import paired_permutation_test

__all__ = [
    "collect_per_query_scores",
    "build_ablation_table",
    "add_significance_markers",
    "format_ablation_report",
]


def _sig_star(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""


def collect_per_query_scores(
    gold_csv: Path,
    model_keys: Optional[list[str]] = None,
    k: int = 10,
    skip_reranker: bool = False,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Collect per-query retrieval scores for all ablation methods."""
    from config import DEFAULT_RERANK_TOP, DEFAULT_TOP_K, EMBEDDING_MODELS
    from embedding_indexing import get_embed_model, load_indices
    from evaluation.evaluation import load_gold_standard, group_gold_by_query, per_query_retrieval_scores
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.hybrid_retriever import HybridRetriever as CompositeHybridRetriever
    from retrieval.reranker import Reranker, RerankedRetriever

    if model_keys is None:
        model_keys = list(EMBEDDING_MODELS.keys())

    reranker = None if skip_reranker else Reranker()

    entries = load_gold_standard(gold_csv)
    query_to_docs = group_gold_by_query(entries)
    queries = sorted(query_to_docs.keys())

    top_k = max(k * 3, DEFAULT_TOP_K, 30)
    rerank_top = max(k, DEFAULT_RERANK_TOP)

    rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        faiss_index, bm25_index, chunks = load_indices(model_key)
        embed_model = get_embed_model(model_key)

        bm25_ret = BM25Retriever(bm25_index, chunks)
        dense_ret = DenseRetriever(faiss_index, chunks, embed_model)
        hybrid_ret = CompositeHybridRetriever(faiss_index, bm25_index, chunks, embed_model)

        retrievers: dict[str, Any] = {
            "bm25": bm25_ret,
            "dense": dense_ret,
            "rrf": hybrid_ret,
        }
        if reranker is not None:
            retrievers["bm25_rerank"] = RerankedRetriever(
                bm25_ret, reranker, initial_k=top_k, final_k=rerank_top
            )
            retrievers["dense_rerank"] = RerankedRetriever(
                dense_ret, reranker, initial_k=top_k, final_k=rerank_top
            )
            retrievers["rrf_rerank"] = RerankedRetriever(
                hybrid_ret, reranker, initial_k=top_k, final_k=rerank_top
            )

        for method_name, retriever in retrievers.items():
            scores = per_query_retrieval_scores(
                retriever,
                gold_path=gold_csv,
                k=k,
                top_k_retrieve=top_k,
                rerank_top=rerank_top,
            )
            for idx, query in enumerate(queries):
                rows.append(
                    {
                        "model_key": model_key,
                        "method": method_name,
                        "query_idx": idx,
                        "query": query,
                        "hit": scores["hit"][idx],
                        "recall": scores["recall"][idx],
                        "mrr": scores["mrr"][idx],
                        "ap": scores["ap"][idx],
                        "ndcg": scores["ndcg"][idx],
                    }
                )

    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


_METHOD_ORDER = ["bm25", "dense", "rrf", "bm25_rerank", "dense_rerank", "rrf_rerank"]


def build_ablation_table(
    metrics_csv: Path,
    k: int = 10,
    metrics: Optional[list[str]] = None,
    dataset: str = "gold_standard",
    level: str = "document",
) -> pd.DataFrame:
    """Build method x (model,metric) ablation table from metrics_all.csv."""
    if metrics is None:
        metrics = ["ndcg", "mrr", "hit_rate"]

    df = pd.read_csv(metrics_csv)
    subset = df[(df["dataset"] == dataset) & (df["level"] == level) & (df["k"] == k)]
    if subset.empty:
        raise ValueError(
            f"No rows for dataset={dataset!r}, level={level!r}, k={k} in {metrics_csv}."
        )

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
    """Annotate ablation table values with significance stars vs baseline."""
    model_keys = per_query_df["model_key"].unique().tolist()
    comparisons: list[tuple[str, str, float]] = []

    for model_key in model_keys:
        baseline_scores = per_query_df[
            (per_query_df["model_key"] == model_key)
            & (per_query_df["method"] == baseline_method)
        ][primary_metric].tolist()
        if not baseline_scores:
            continue

        methods = per_query_df[per_query_df["model_key"] == model_key]["method"].unique()
        for method in methods:
            if method == baseline_method:
                continue
            method_scores = per_query_df[
                (per_query_df["model_key"] == model_key)
                & (per_query_df["method"] == method)
            ][primary_metric].tolist()
            if len(method_scores) != len(baseline_scores):
                continue
            p = paired_permutation_test(
                method_scores,
                baseline_scores,
                n_permutations=n_permutations,
                rng_seed=rng_seed,
            )
            comparisons.append((model_key, method, p))

    sig: dict[tuple[str, str], str] = {}
    for model_key, method, p_val in comparisons:
        sig[(model_key, method)] = _sig_star(float(p_val), alpha)

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
    """Format ablation table as plain-text report."""
    lines = ["", "=" * 100, f"  {title}", "=" * 100]

    def _extra_row(label: str, metrics: dict[int, RetrievalMetrics]) -> str:
        if k not in metrics:
            return ""
        m = metrics[k]
        return (
            f"  {label:<28}  NDCG={m.ndcg:.3f}  MRR={m.mrr:.3f}  Hit={m.hit_rate:.3f}"
            f"  (all models identical)"
        )

    if random_metrics:
        lines.append(_extra_row("Random baseline", random_metrics))
    if oracle_metrics:
        lines.append(_extra_row("Oracle upper bound", oracle_metrics))
    if random_metrics or oracle_metrics:
        lines.append("-" * 100)

    lines.append(ablation_df.to_string())
    lines.append("")
    lines.append("  * p<0.05   ** p<0.01   *** p<0.001  (paired permutation test)")
    lines.append("=" * 100)
    return "\n".join(lines)
