"""Shared helpers for unified and robustness evaluation commands."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Any

from config import INDEX_DIR, RRF_K
from embedding_indexing import get_embed_model, load_indices
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever as CompositeHybridRetriever
from retrieval.reranker import RerankedRetriever, Reranker


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(message: str) -> None:
    print(f"[{_ts()}] [mteb] {message}", flush=True)

def _safe_retrieve(retriever: Any, query: str, top_k: int):
    try:
        return retriever.retrieve(query, top_k=top_k)
    except TypeError:
        return retriever.retrieve(query)


def _paired_effect_size_dz(scores_a: list[float], scores_b: list[float]) -> float:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    if len(diff) < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    sorted_p = [float(p_values[i]) for i in order]
    adjusted_sorted: list[float] = [0.0] * m
    prev = 0.0
    for i, p in enumerate(sorted_p):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)
        adjusted_sorted[i] = adj
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


def _build_metrics_summary_tables(metrics_df: pd.DataFrame, k_for_summary: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                second_model = ""
                second_value = float("nan")
                gap = float("nan")
            comp_rows.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "method": method,
                    "k": int(k_for_summary),
                    "metric": metric,
                    "best_model": str(best["model_key"]),
                    "best_value": float(best[metric]),
                    "second_model": second_model,
                    "second_value": second_value,
                    "gap_to_second": gap,
                    "num_models_compared": int(len(ranked)),
                }
            )
    comparison_df = pd.DataFrame(
        comp_rows,
        columns=[
            "dataset",
            "level",
            "method",
            "k",
            "metric",
            "best_model",
            "best_value",
            "second_model",
            "second_value",
            "gap_to_second",
            "num_models_compared",
        ],
    )
    return summary_df, comparison_df


def _validate_ranking_consistency(metrics_df: pd.DataFrame, ranking_df: pd.DataFrame, k_for_ranking: int = 10) -> None:
    expected = (
        metrics_df[metrics_df["k"] == k_for_ranking]
        .sort_values(
            ["dataset", "level", "ndcg", "model_key", "method"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    actual = (
        ranking_df.copy()
        .sort_values(
            ["dataset", "level", "ndcg", "model_key", "method"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if expected.shape != ranking_df.shape:
        raise RuntimeError(
            "Ranking export shape mismatch with evaluated metrics table. "
            f"expected={expected.shape}, actual={ranking_df.shape}"
        )
    if list(expected.columns) != list(actual.columns):
        raise RuntimeError(
            "Ranking export columns mismatch with evaluated metrics table. "
            f"expected={list(expected.columns)}, actual={list(actual.columns)}"
        )

    try:
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Ranking export does not match evaluated and sorted k=10 metrics. "
            f"Details: {exc}"
        ) from exc


def _metrics_to_rows(
    metrics_by_k: dict[int, object],
    *,
    dataset: str,
    level: str,
    model_key: str,
    method: str,
) -> list[dict]:
    rows: list[dict] = []
    for k, m in sorted(metrics_by_k.items()):
        rows.append(
            {
                "dataset": dataset,
                "level": level,
                "model_key": model_key,
                "method": method,
                "k": k,
                "hit_rate": m.hit_rate,
                "recall": m.recall,
                "precision": m.precision,
                "mrr": m.mrr,
                "map": m.map_score,
                "ndcg": m.ndcg,
                "mean_rank": m.mean_rank,
                "chunk_hit_rate": m.chunk_hit_rate,
                "num_queries": m.num_queries,
            }
        )
    return rows


def _indices_exist(model_key: str) -> bool:
    prefix = INDEX_DIR / model_key
    return (
        Path(str(prefix) + "_faiss.index").exists()
        and Path(str(prefix) + "_bm25.pkl").exists()
        and Path(str(prefix) + "_chunks.pkl").exists()
    )


def _build_retrievers_for_model(
    model_key: str,
    reranker: Reranker | None,
    top_k: int,
    rerank_top: int,
    rrf_k: int = RRF_K,
) -> dict[str, Any]:
    faiss_index, bm25_index, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)

    bm25 = BM25Retriever(bm25_index, chunks)
    dense = DenseRetriever(faiss_index, chunks, embed_model)
    hybrid = CompositeHybridRetriever(faiss_index, bm25_index, chunks, embed_model, rrf_k=rrf_k)

    retrievers: dict[str, Any] = {
        "bm25": bm25,
        "dense": dense,
        "rrf": hybrid,
    }
    if reranker is not None:
        retrievers["bm25_rerank"] = RerankedRetriever(
            bm25, reranker, initial_k=max(top_k * 2, 30), final_k=rerank_top
        )
        retrievers["dense_rerank"] = RerankedRetriever(
            dense, reranker, initial_k=max(top_k * 2, 30), final_k=rerank_top
        )
        retrievers["rrf_rerank"] = RerankedRetriever(
            hybrid, reranker, initial_k=max(top_k * 2, 30), final_k=rerank_top
        )
    return retrievers
