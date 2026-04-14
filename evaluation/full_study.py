"""Thesis-focused evaluation orchestration and analysis helpers.

This script provides two reproducible study entry points:
1) retrieval-study: robust retrieval ablation + baseline comparison
2) prompt-study: classification + judge result analysis

It reuses existing pipeline evaluators so behavior stays consistent with
`main.py evaluate` and `main.py prompt` outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Allow running as `python evaluation/full_study.py ...` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    EMBEDDING_MODELS,
    EVAL_K_VALUES,
    EVIDENCE_CSV,
    GOLD_STANDARD_CSV,
    OUTPUT_DIR,
    SPLADE_MAX_LENGTH,
    SPLADE_MODEL,
    WHITEPAPER_RECOMMENDATIONS_CSV,
)
from evaluation.experiment_commands import cmd_robustness, cmd_unified_eval

DEFAULT_BASELINE_MODELS = ["bge-m3", "e5-large-v2", "e5-mistral"]
DEFAULT_K_VALUES = [1, 3, 5, 10, 20]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thesis evaluation orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ret = sub.add_parser("retrieval-study", help="Run robust retrieval ablation and baseline comparison")
    p_ret.add_argument("--models", nargs="+", default=DEFAULT_BASELINE_MODELS)
    p_ret.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "eval_thesis")
    p_ret.add_argument("--old-metrics-csv", type=Path, default=OUTPUT_DIR / "eval_unified_old" / "metrics_all.csv")
    p_ret.add_argument("--ranking-k", type=int, default=10)
    p_ret.add_argument("--skip-mteb", action="store_true")
    p_ret.add_argument("--skip-reranker", action="store_true")
    p_ret.add_argument("--auto-build-indices", action="store_true")
    p_ret.add_argument("--include-splade", action="store_true")
    p_ret.add_argument("--force-cpu", action="store_true")
    p_ret.add_argument("--with-robustness-all-models", action="store_true")
    p_ret.add_argument("--robust-k", type=int, default=10)

    p_prompt = sub.add_parser("prompt-study", help="Analyze prompt classification and judge outputs")
    p_prompt.add_argument("--prompt-csv", type=Path, required=True)
    p_prompt.add_argument("--judge-csv", type=Path, default=None)
    p_prompt.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "eval_prompt")

    p_k = sub.add_parser("k-compare", help="Compare k values using an existing metrics_all.csv")
    p_k.add_argument("--metrics-csv", type=Path, default=OUTPUT_DIR / "eval_unified_old" / "metrics_all.csv")
    p_k.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    p_k.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "eval_k_compare")

    return parser


DEFAULT_STUDY_TOP_K = 20
DEFAULT_STUDY_RERANK_TOP = 10
DEFAULT_STUDY_EXPORT_K = 10


def _filter_available_models(models: list[str]) -> list[str]:
    available = [m for m in models if m in EMBEDDING_MODELS]
    missing = [m for m in models if m not in EMBEDDING_MODELS]
    if missing:
        print(f"[warn] Skipping unknown model keys: {missing}")
    if not available:
        raise ValueError("No valid embedding models after filtering.")
    return available


def _build_unified_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        models=_filter_available_models(args.models),
        gold_csv=Path(GOLD_STANDARD_CSV),
        whitepaper_csv=Path(WHITEPAPER_RECOMMENDATIONS_CSV),
        output_dir=Path(args.output_dir),
        top_k=DEFAULT_STUDY_TOP_K,
        rerank_top=DEFAULT_STUDY_RERANK_TOP,
        export_k=DEFAULT_STUDY_EXPORT_K,
        k_values=sorted(set(EVAL_K_VALUES)),
        # mteb_dataset="mteb/legalbench_consumer_contracts_qa",
        mteb_dataset="mteb/MuPLeR-retrieval",
        mteb_split="test",
        max_corpus=None,
        full_mteb=False,
        skip_whitepaper=False,
        skip_mteb=bool(args.skip_mteb),
        skip_reranker=bool(args.skip_reranker),
        auto_build_indices=bool(args.auto_build_indices),
        evidence_csv=Path(EVIDENCE_CSV),
        include_splade=bool(args.include_splade),
        splade_model=SPLADE_MODEL,
        splade_max_length=SPLADE_MAX_LENGTH,
        force_cpu=bool(args.force_cpu),
        with_robustness=False,
    )


def _run_robustness_for_models(
    *,
    models: list[str],
    gold_csv: Path,
    output_dir: Path,
    top_k: int,
    rerank_top: int,
    robust_k: int,
    skip_reranker: bool,
) -> None:
    for model in models:
        print(f"[robustness] Running per-query robustness for model={model}")
        robust_args = argparse.Namespace(
            model=model,
            gold_csv=gold_csv,
            k=robust_k,
            top_k=top_k,
            rerank_top=rerank_top,
            output_dir=output_dir / "robustness",
            skip_reranker=skip_reranker,
        )
        cmd_robustness(robust_args)


def _build_selection_tables(metrics_csv: Path, out_dir: Path, ranking_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metrics_csv)
    df = df[df["k"] == ranking_k].copy()

    best = (
        df.sort_values(["dataset", "level", "ndcg"], ascending=[True, True, False])
        .groupby(["dataset", "level"], as_index=False)
        .first()
        .rename(
            columns={
                "model_key": "best_model",
                "method": "best_method",
                "ndcg": "best_ndcg",
                "mrr": "best_mrr",
                "hit_rate": "best_hit_rate",
            }
        )
    )

    # Add second-best competitor and margin.
    second_rows: list[dict[str, Any]] = []
    for (dataset, level), grp in df.groupby(["dataset", "level"], dropna=False):
        ranked = grp.sort_values("ndcg", ascending=False).reset_index(drop=True)
        row: dict[str, Any] = {
            "dataset": dataset,
            "level": level,
            "second_model": "",
            "second_method": "",
            "second_ndcg": np.nan,
            "ndcg_gap": np.nan,
        }
        if len(ranked) > 1:
            row["second_model"] = str(ranked.iloc[1]["model_key"])
            row["second_method"] = str(ranked.iloc[1]["method"])
            row["second_ndcg"] = float(ranked.iloc[1]["ndcg"])
            row["ndcg_gap"] = float(ranked.iloc[0]["ndcg"] - ranked.iloc[1]["ndcg"])
        second_rows.append(row)

    second_df = pd.DataFrame(second_rows)
    selection_df = best.merge(second_df, on=["dataset", "level"], how="left")
    selection_df = selection_df.sort_values(["dataset", "level"]).reset_index(drop=True)

    selection_csv = out_dir / "best_model_method_selection_k10.csv"
    selection_df.to_csv(selection_csv, index=False)

    # Embedding-only view: best average by method + model across datasets.
    aggregate = (
        df.groupby(["model_key", "method"], as_index=False)[["ndcg", "mrr", "hit_rate"]]
        .mean()
        .sort_values("ndcg", ascending=False)
        .reset_index(drop=True)
    )
    aggregate_csv = out_dir / "global_ranking_by_mean_metrics_k10.csv"
    aggregate.to_csv(aggregate_csv, index=False)

    print(f"[done] Selection table -> {selection_csv}")
    print(f"[done] Global ranking -> {aggregate_csv}")
    return selection_df, aggregate


def _compare_against_old(old_metrics_csv: Path, new_metrics_csv: Path, out_dir: Path, k: int) -> pd.DataFrame:
    old_df = pd.read_csv(old_metrics_csv)
    new_df = pd.read_csv(new_metrics_csv)

    key_cols = ["dataset", "level", "model_key", "method", "k"]
    metric_cols = ["hit_rate", "mrr", "ndcg"]

    old_k = old_df[old_df["k"] == k][key_cols + metric_cols].copy()
    new_k = new_df[new_df["k"] == k][key_cols + metric_cols].copy()

    merged = old_k.merge(new_k, on=key_cols, suffixes=("_old", "_new"), how="inner")
    if merged.empty:
        print("[warn] No overlapping rows between old and new metrics for delta comparison.")
        return merged

    for m in metric_cols:
        merged[f"delta_{m}"] = merged[f"{m}_new"] - merged[f"{m}_old"]

    merged = merged.sort_values(["dataset", "level", "delta_ndcg"], ascending=[True, True, False])
    out_csv = out_dir / "baseline_vs_current_delta_k10.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[done] Baseline comparison -> {out_csv}")
    return merged


def run_retrieval_study(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unified_args = _build_unified_args(args)
    print("[retrieval-study] Running unified retrieval evaluation...")
    cmd_unified_eval(unified_args)

    if args.with_robustness_all_models:
        _run_robustness_for_models(
            models=unified_args.models,
            gold_csv=Path(unified_args.gold_csv),
            output_dir=out_dir,
            top_k=unified_args.top_k,
            rerank_top=unified_args.rerank_top,
            robust_k=args.robust_k,
            skip_reranker=unified_args.skip_reranker,
        )

    metrics_csv = out_dir / "metrics_all.csv"
    selection_df, global_df = _build_selection_tables(metrics_csv, out_dir, ranking_k=args.ranking_k)

    delta_df = pd.DataFrame()
    if args.old_metrics_csv:
        delta_df = _compare_against_old(
            old_metrics_csv=Path(args.old_metrics_csv),
            new_metrics_csv=metrics_csv,
            out_dir=out_dir,
            k=args.ranking_k,
        )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": unified_args.models,
        "output_dir": str(out_dir),
        "ranking_k": args.ranking_k,
        "selected_best_by_dataset_level": selection_df.to_dict(orient="records"),
        "global_ranking_top5": global_df.head(5).to_dict(orient="records"),
        "delta_rows": int(len(delta_df)),
    }
    summary_json = out_dir / "thesis_retrieval_study_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] Retrieval study summary -> {summary_json}")


def run_k_compare(args: argparse.Namespace) -> None:
    metrics_csv = Path(args.metrics_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    required = {"dataset", "level", "k", "model_key", "method", "hit_rate", "mrr", "ndcg"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metrics CSV: {sorted(missing)}")

    k_values = [int(k) for k in args.k_values]
    df = df[df["k"].isin(k_values)].copy()
    if df.empty:
        raise ValueError("No rows found for the requested k values.")

    # Average trend across all evaluated configurations.
    avg_trend = (
        df.groupby("k", as_index=False)[["ndcg", "mrr", "hit_rate"]]
        .mean()
        .rename(columns={"ndcg": "mean_ndcg", "mrr": "mean_mrr", "hit_rate": "mean_hit_rate"})
        .sort_values("k")
        .reset_index(drop=True)
    )
    avg_trend["rank_by_ndcg"] = avg_trend["mean_ndcg"].rank(ascending=False, method="min").astype(int)

    # Best configuration for each k per dataset/level.
    winners = (
        df.sort_values(["dataset", "level", "k", "ndcg"], ascending=[True, True, True, False])
        .groupby(["dataset", "level", "k"], as_index=False)
        .first()
        .rename(
            columns={
                "model_key": "best_model",
                "method": "best_method",
                "ndcg": "best_ndcg",
                "mrr": "best_mrr",
                "hit_rate": "best_hit_rate",
            }
        )
    )

    avg_csv = out_dir / "k_comparison_average_trend.csv"
    winners_csv = out_dir / "k_comparison_winners_by_dataset_level.csv"
    avg_trend.to_csv(avg_csv, index=False)
    winners.to_csv(winners_csv, index=False)

    best_k_row = avg_trend.sort_values("mean_ndcg", ascending=False).iloc[0]
    summary = {
        "metrics_csv": str(metrics_csv),
        "k_values": k_values,
        "best_k_by_mean_ndcg": int(best_k_row["k"]),
        "best_k_mean_ndcg": float(best_k_row["mean_ndcg"]),
        "best_k_mean_mrr": float(best_k_row["mean_mrr"]),
        "best_k_mean_hit_rate": float(best_k_row["mean_hit_rate"]),
        "outputs": {
            "average_trend_csv": str(avg_csv),
            "winners_csv": str(winners_csv),
        },
    }

    summary_json = out_dir / "k_comparison_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] K comparison average trend -> {avg_csv}")
    print(f"[done] K comparison winners -> {winners_csv}")
    print(f"[done] K comparison summary -> {summary_json}")


def _split_semicolon_values(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    values = cleaned.str.split(";")
    flattened = [item.strip() for sub in values for item in sub if item and item.strip()]
    return pd.Series(flattened, dtype="string")


def run_prompt_study(args: argparse.Namespace) -> None:
    prompt_csv = Path(args.prompt_csv)
    judge_csv = Path(args.judge_csv) if args.judge_csv else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_df = pd.read_csv(prompt_csv)

    prompt_df["alignment_label"] = prompt_df.get("alignment_label", "").fillna("").astype(str)
    prompt_df["human_label"] = prompt_df.get("human_label", "").fillna("").astype(str)

    label_counts = (
        prompt_df["alignment_label"]
        .replace("", "<empty>")
        .value_counts(dropna=False)
        .rename_axis("alignment_label")
        .reset_index(name="count")
    )
    label_counts.to_csv(out_dir / "classification_label_distribution.csv", index=False)

    mode_counts = (
        prompt_df.get("retrieval_mode", pd.Series(["unknown"] * len(prompt_df)))
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .rename_axis("retrieval_mode")
        .reset_index(name="count")
    )
    mode_counts.to_csv(out_dir / "classification_retrieval_mode_distribution.csv", index=False)

    cited_chunks = _split_semicolon_values(prompt_df.get("cited_chunk_ids", pd.Series(dtype="string")))
    cited_chunk_counts = cited_chunks.value_counts().rename_axis("chunk_id").reset_index(name="citations")
    cited_chunk_counts.to_csv(out_dir / "classification_cited_chunk_frequency.csv", index=False)

    report: dict[str, Any] = {
        "num_rows": int(len(prompt_df)),
        "num_non_empty_labels": int((prompt_df["alignment_label"] != "").sum()),
        "num_human_labels": int((prompt_df["human_label"] != "").sum()),
        "judge_csv_input": str(judge_csv) if judge_csv is not None else None,
        "judge_csv_exists": bool(judge_csv.exists()) if judge_csv is not None else False,
    }

    eval_mask = (prompt_df["alignment_label"] != "") & (prompt_df["human_label"] != "")
    eval_df = prompt_df[eval_mask].copy()
    if not eval_df.empty:
        y_true = eval_df["human_label"].tolist()
        y_pred = eval_df["alignment_label"].tolist()
        report["agreement_accuracy"] = float(accuracy_score(y_true, y_pred))
        report["agreement_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        report["agreement_weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(cls_report).transpose().to_csv(out_dir / "classification_report_vs_human.csv", index=True)

    if judge_csv is not None and judge_csv.exists():
        judge_df = pd.read_csv(judge_csv)
        judge_summary = {
            "num_rows": int(len(judge_df)),
            "label_score_mean": float(judge_df["label_score"].mean()),
            "justification_score_mean": float(judge_df["justification_score"].mean()),
            "evidence_score_mean": float(judge_df["evidence_score"].mean()),
            "overall_score_mean": float(judge_df["overall_score"].mean()),
            "overall_score_std": float(judge_df["overall_score"].std(ddof=0)),
        }
        report["judge_summary"] = judge_summary

        bins = [0, 2, 3, 4, 5]
        labels = ["very_weak", "weak", "good", "strong"]
        judge_df["overall_band"] = pd.cut(judge_df["overall_score"], bins=bins, labels=labels, include_lowest=True)
        judge_df["overall_band"].value_counts(dropna=False).rename_axis("overall_band").reset_index(
            name="count"
        ).to_csv(out_dir / "judge_overall_band_distribution.csv", index=False)
    elif judge_csv is not None:
        print(f"[warn] --judge-csv was provided but file does not exist: {judge_csv}")
        print("[warn] prompt-study does not generate a raw judge CSV; pass an existing one from `main.py prompt --judge`.")
        report["judge_summary_warning"] = "judge_csv_not_found"

    summary_json = out_dir / "thesis_prompt_study_summary.json"
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] Prompt study summary -> {summary_json}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "retrieval-study":
        run_retrieval_study(args)
    elif args.command == "prompt-study":
        run_prompt_study(args)
    elif args.command == "k-compare":
        run_k_compare(args)


if __name__ == "__main__":
    main()
