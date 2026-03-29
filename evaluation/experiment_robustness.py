"""Robustness command for retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import INDEX_DIR
from evaluation.evaluation import group_gold_by_query, load_gold_standard, per_query_retrieval_scores
from evaluation.metrics import bootstrap_ci, paired_permutation_test
from evaluation.experiment_helpers import (
    _build_retrievers_for_model,
    _effect_size_label,
    _holm_bonferroni,
    _paired_effect_size_dz,
)
from retrieval.reranker import Reranker


def _mean_or_inf(arr: list[float]) -> float:
    finite = [x for x in arr if x != float("inf")]
    return float(np.mean(finite)) if finite else float("inf")


def _error_categories(
    retriever: Any,
    gold_csv: Path,
    k_main: int,
    top_k: int,
    rerank_top: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = load_gold_standard(gold_csv)
    query_to_docs = group_gold_by_query(entries)

    rows: list[dict[str, Any]] = []
    max_probe = max(20, top_k)

    for query, relevant in query_to_docs.items():
        try:
            result = retriever.retrieve(query, top_k=max_probe * 3, rerank_top=max_probe)
        except TypeError:
            result = retriever.retrieve(query, top_k=max_probe)

        seen = []
        seen_set = set()
        for chunk in result.ranked_chunks:
            doc = chunk.document
            if doc not in seen_set:
                seen.append(doc)
                seen_set.add(doc)

        topk = set(seen[:k_main])
        top20 = set(seen[:20])
        rel = set(relevant)

        n_topk = len(rel & topk)
        if n_topk == len(rel):
            category = "success"
        elif n_topk > 0:
            category = "partial"
        elif rel & top20:
            category = "late_hit"
        else:
            category = "complete_miss"

        rows.append(
            {
                "query": query,
                "relevant_docs": "; ".join(sorted(rel)),
                "retrieved_top10_docs": "; ".join(seen[:10]),
                "category": category,
            }
        )

    per_query_df = pd.DataFrame(rows)
    counts_df = (
        per_query_df["category"].value_counts(dropna=False).rename_axis("category").reset_index(name="count")
    )
    counts_df["share"] = counts_df["count"] / max(len(per_query_df), 1)
    return per_query_df, counts_df


def cmd_robustness(args: argparse.Namespace) -> None:
    """Run publishability-focused robustness analysis."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    idx_prefix = INDEX_DIR / args.model
    if not (Path(str(idx_prefix) + "_faiss.index").exists() and Path(str(idx_prefix) + "_bm25.pkl").exists()):
        raise FileNotFoundError(
            f"Indices for model '{args.model}' are missing. Build first with: "
            f"python main.py build -i outputs/evidence.csv -m {args.model}"
        )

    reranker = None
    if not args.skip_reranker:
        print("[setup] Loading cross-encoder reranker...")
        try:
            reranker = Reranker()
        except Exception as exc:
            msg = str(exc).lower()
            if "outofmemory" in msg or "cuda out of memory" in msg:
                print("[warn] CUDA OOM while loading reranker; continuing with --skip-reranker behavior.")
                reranker = None
            else:
                raise

    retrievers = _build_retrievers_for_model(
        args.model,
        top_k=args.top_k,
        rerank_top=args.rerank_top,
        reranker=reranker,
    )

    gold_queries = sorted(group_gold_by_query(load_gold_standard(args.gold_csv)).keys())

    per_query_long_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []

    for method_name, retriever in retrievers.items():
        scores = per_query_retrieval_scores(
            retriever,
            gold_path=args.gold_csv,
            k=args.k,
            top_k_retrieve=max(args.top_k * 3, 30),
            rerank_top=args.rerank_top,
            level="document",
        )

        n = len(scores["hit"])
        if n != len(gold_queries):
            raise RuntimeError(
                "Per-query score length mismatch with gold query set: "
                f"scores={n}, gold_queries={len(gold_queries)}"
            )
        for i in range(n):
            per_query_long_rows.append(
                {
                    "model_key": args.model,
                    "method": method_name,
                    "query_index": i,
                    "query": gold_queries[i],
                    "hit": float(scores["hit"][i]),
                    "recall": float(scores["recall"][i]),
                    "precision": float(scores["precision"][i]),
                    "mrr": float(scores["mrr"][i]),
                    "ap": float(scores["ap"][i]),
                    "ndcg": float(scores["ndcg"][i]),
                    "rank": float(scores["rank"][i]),
                }
            )

        for metric in ("hit", "mrr", "ndcg"):
            mean_val, ci_lo, ci_hi = bootstrap_ci(scores[metric], n_bootstrap=10_000, confidence=0.95, rng_seed=42)
            ci_rows.append(
                {
                    "model_key": args.model,
                    "method": method_name,
                    "k": args.k,
                    "metric": metric,
                    "mean": mean_val,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "num_queries": len(scores[metric]),
                }
            )

        ci_rows.append(
            {
                "model_key": args.model,
                "method": method_name,
                "k": args.k,
                "metric": "mean_rank",
                "mean": _mean_or_inf(scores["rank"]),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "num_queries": len(scores["rank"]),
            }
        )

    per_query_df = pd.DataFrame(per_query_long_rows)
    per_query_csv = args.output_dir / f"per_query_scores_k{args.k}_{args.model}.csv"
    per_query_df.to_csv(per_query_csv, index=False)

    ci_df = pd.DataFrame(ci_rows)
    ci_csv = args.output_dir / f"ci_table_k{args.k}_{args.model}.csv"
    ci_df.to_csv(ci_csv, index=False)

    pair_rows: list[dict[str, Any]] = []
    methods = sorted(per_query_df["method"].unique().tolist())
    for a, b in combinations(methods, 2):
        dfa = per_query_df[per_query_df["method"] == a].sort_values("query_index").reset_index(drop=True)
        dfb = per_query_df[per_query_df["method"] == b].sort_values("query_index").reset_index(drop=True)
        if not dfa["query_index"].equals(dfb["query_index"]):
            raise RuntimeError(f"Query index mismatch for paired comparison: {a} vs {b}")
        if not dfa["query"].equals(dfb["query"]):
            raise RuntimeError(f"Query text mismatch for paired comparison: {a} vs {b}")
        for metric in ("mrr", "ndcg", "hit"):
            scores_a = dfa[metric].astype(float).tolist()
            scores_b = dfb[metric].astype(float).tolist()
            p_val = paired_permutation_test(
                scores_a,
                scores_b,
                n_permutations=10_000,
                rng_seed=42,
            )
            delta = float(dfa[metric].mean() - dfb[metric].mean())
            effect_dz = _paired_effect_size_dz(scores_a, scores_b)
            pair_rows.append(
                {
                    "model_key": args.model,
                    "k": args.k,
                    "metric": metric,
                    "method_a": a,
                    "method_b": b,
                    "delta_mean": delta,
                    "p_value": p_val,
                    "effect_size_dz": effect_dz,
                    "effect_label": _effect_size_label(effect_dz),
                }
            )

    pvals_df = pd.DataFrame(pair_rows)
    if not pvals_df.empty:
        pvals_df["p_value_holm"] = np.nan
        pvals_df["significant_holm_0_05"] = False
        for metric, group in pvals_df.groupby("metric", sort=False):
            adjusted = _holm_bonferroni(group["p_value"].astype(float).tolist())
            pvals_df.loc[group.index, "p_value_holm"] = adjusted
            pvals_df.loc[group.index, "significant_holm_0_05"] = [float(x) < 0.05 for x in adjusted]
    pvals_csv = args.output_dir / f"pairwise_permutation_k{args.k}_{args.model}.csv"
    pvals_df.to_csv(pvals_csv, index=False)

    comparison_report_csv = args.output_dir / f"comparison_report_k{args.k}_{args.model}.csv"
    comparison_report_df = (
        pvals_df.sort_values(["metric", "p_value_holm", "p_value", "delta_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    comparison_report_df.to_csv(comparison_report_csv, index=False)

    mean_by_method_metric = (
        per_query_df.groupby("method")[["hit", "mrr", "ndcg", "rank"]].mean().to_dict("index")
    )
    ablations = [
        ("bm25", "bm25_rerank", "reranker_gain_bm25"),
        ("dense", "dense_rerank", "reranker_gain_dense"),
        ("rrf", "rrf_rerank", "reranker_gain_rrf"),
        ("dense", "rrf", "hybrid_vs_dense"),
        ("bm25", "rrf", "hybrid_vs_bm25"),
    ]
    abl_rows: list[dict[str, Any]] = []
    for base, variant, name in ablations:
        if base not in mean_by_method_metric or variant not in mean_by_method_metric:
            continue
        b = mean_by_method_metric[base]
        v = mean_by_method_metric[variant]
        abl_rows.append(
            {
                "model_key": args.model,
                "ablation": name,
                "base": base,
                "variant": variant,
                "delta_hit": float(v["hit"] - b["hit"]),
                "delta_mrr": float(v["mrr"] - b["mrr"]),
                "delta_ndcg": float(v["ndcg"] - b["ndcg"]),
                "delta_mean_rank": float(v["rank"] - b["rank"]),
            }
        )

    ablation_df = pd.DataFrame(abl_rows)
    ablation_csv = args.output_dir / f"ablation_deltas_k{args.k}_{args.model}.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    ndcg_means = per_query_df.groupby("method")["ndcg"].mean().sort_values(ascending=False)
    best_method = ndcg_means.index[0]
    best_retriever = retrievers[best_method]
    per_query_error_df, error_counts_df = _error_categories(
        best_retriever,
        args.gold_csv,
        k_main=args.k,
        top_k=args.top_k,
        rerank_top=args.rerank_top,
    )

    errors_csv = args.output_dir / f"error_analysis_queries_k{args.k}_{args.model}_{best_method}.csv"
    errors_counts_csv = args.output_dir / f"error_analysis_summary_k{args.k}_{args.model}_{best_method}.csv"
    per_query_error_df.to_csv(errors_csv, index=False)
    error_counts_df.to_csv(errors_counts_csv, index=False)

    negative_csv = args.output_dir / f"negative_cases_k{args.k}_{args.model}_{best_method}.csv"
    per_query_error_df[per_query_error_df["category"].isin(["late_hit", "complete_miss"])].to_csv(
        negative_csv, index=False
    )

    interpretation_txt = args.output_dir / f"robustness_interpretation_k{args.k}_{args.model}.txt"
    interp_lines = [
        f"Robustness interpretation for model={args.model}, k={args.k}",
        "",
        "Method means (hit, mrr, ndcg, rank):",
    ]
    means_table = per_query_df.groupby("method")[["hit", "mrr", "ndcg", "rank"]].mean().sort_values("ndcg", ascending=False)
    for method_name, row in means_table.iterrows():
        interp_lines.append(
            f"- {method_name}: hit={row['hit']:.4f}, mrr={row['mrr']:.4f}, ndcg={row['ndcg']:.4f}, rank={row['rank']:.2f}"
        )
    if not comparison_report_df.empty:
        interp_lines.append("")
        interp_lines.append("Paired differences significant after Holm correction (alpha=0.05):")
        sig_df = comparison_report_df[comparison_report_df["significant_holm_0_05"]]
        if sig_df.empty:
            interp_lines.append("- None")
        else:
            for _, row in sig_df.iterrows():
                interp_lines.append(
                    f"- {row['metric']}: {row['method_a']} vs {row['method_b']} "
                    f"(delta={row['delta_mean']:.4f}, p_holm={row['p_value_holm']:.4g}, effect={row['effect_label']}, dz={row['effect_size_dz']:.3f})"
                )
    interpretation_txt.write_text("\n".join(interp_lines) + "\n", encoding="utf-8")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "k": args.k,
        "best_method_by_ndcg": best_method,
        "outputs": {
            "per_query_scores": str(per_query_csv),
            "ci_table": str(ci_csv),
            "pairwise_permutation": str(pvals_csv),
            "comparison_report": str(comparison_report_csv),
            "ablation_deltas": str(ablation_csv),
            "error_queries": str(errors_csv),
            "error_summary": str(errors_counts_csv),
            "negative_cases": str(negative_csv),
            "interpretation": str(interpretation_txt),
        },
    }
    with open(args.output_dir / f"robustness_summary_k{args.k}_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] robustness analysis complete")
