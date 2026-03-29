"""CLI command handlers for evaluation workflows (thin entrypoints)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import EMBEDDING_MODELS, JUDGE_MODEL, LLM_MODEL, RERANKER_MODEL
from evaluation.full_eval import build_ablation_table, format_ablation_report
from evaluation.experiment_helpers import _build_metrics_summary_tables
from evaluation.experiment_unified import cmd_unified_eval
from evaluation.experiment_robustness import cmd_robustness

def cmd_merge_eval(args: argparse.Namespace) -> None:
    """Merge one or more remote metrics CSVs into local unified outputs."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_metrics_csv = args.output_dir / "metrics_all.csv"
    frames: list[pd.DataFrame] = []

    if local_metrics_csv.exists():
        frames.append(pd.read_csv(local_metrics_csv))

    for remote_csv in args.remote_csv:
        remote_path = Path(remote_csv)
        if not remote_path.exists():
            raise FileNotFoundError(f"Remote metrics CSV not found: {remote_path}")
        frames.append(pd.read_csv(remote_path))

    if not frames:
        raise FileNotFoundError(
            "No metrics CSV available to merge. Run `main.py evaluate` first or provide --remote-csv."
        )

    metrics_df = pd.concat(frames, ignore_index=True)
    required_cols = {"dataset", "level", "model_key", "method", "k", "ndcg", "mrr", "hit_rate"}
    missing = required_cols.difference(metrics_df.columns)
    if missing:
        raise ValueError(f"Merged metrics missing required columns: {sorted(missing)}")

    dedup_keys = ["dataset", "level", "model_key", "method", "k"]
    metrics_df = metrics_df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    metrics_df.to_csv(local_metrics_csv, index=False)

    ranking_k = int(getattr(args, "ranking_k", 10))
    ranking_source = metrics_df[metrics_df["k"] == ranking_k]
    if ranking_source.empty:
        ranking_k = int(metrics_df["k"].max())
        ranking_source = metrics_df[metrics_df["k"] == ranking_k]

    ranking_df = (
        ranking_source
        .sort_values(["dataset", "level", "ndcg"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    ranking_csv = args.output_dir / "ranking_k10.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    summary_k_df, comparison_k_df = _build_metrics_summary_tables(metrics_df, k_for_summary=ranking_k)
    summary_csv = args.output_dir / "metrics_summary_k10.csv"
    comparison_csv = args.output_dir / "comparison_k10.csv"
    summary_k_df.to_csv(summary_csv, index=False)
    comparison_k_df.to_csv(comparison_csv, index=False)

    interpretation_lines: list[str] = [
        f"Unified Evaluation Interpretation (k={ranking_k})",
        "",
        f"Total result rows: {len(summary_k_df)}",
        "Top systems by dataset/level (NDCG):",
    ]
    top_by_group = (
        summary_k_df.sort_values("ndcg", ascending=False)
        .groupby(["dataset", "level"], as_index=False)
        .first()
    )
    for _, row in top_by_group.iterrows():
        interpretation_lines.append(
            f"- {row['dataset']} | {row['level']}: {row['model_key']} + {row['method']} "
            f"(NDCG={row['ndcg']:.4f}, MRR={row['mrr']:.4f}, Hit={row['hit_rate']:.4f})"
        )
    interpretation_txt = args.output_dir / "interpretation_k10.txt"
    interpretation_txt.write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    try:
        ablation_df = build_ablation_table(local_metrics_csv, k=ranking_k)
        ablation_csv = args.output_dir / "ablation_table.csv"
        ablation_txt = args.output_dir / "ablation_table.txt"
        ablation_df.to_csv(ablation_csv)
        ablation_txt.write_text(format_ablation_report(ablation_df, k=ranking_k), encoding="utf-8")
    except Exception as exc:
        print(f"[warn] Could not regenerate ablation artifacts during merge: {exc}")

    print("\n[done] Merged evaluation metrics.")
    print(f"[done] Metrics: {local_metrics_csv}")
    print(f"[done] Ranking: {ranking_csv}")
    print(f"[done] Summary: {summary_csv}")


def cmd_download_models(args: argparse.Namespace) -> None:
    """Pre-download embedding/reranker/LLM models."""
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for key in args.embedding_models:
        if key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model key: {key}")
        model_name = EMBEDDING_MODELS[key]
        print(f"[download] SentenceTransformer: {model_name}")
        SentenceTransformer(model_name)

    print(f"[download] CrossEncoder: {RERANKER_MODEL}")
    CrossEncoder(RERANKER_MODEL)

    if args.include_llms:
        for model_name in (LLM_MODEL, JUDGE_MODEL):
            print(f"[download] Tokenizer: {model_name}")
            AutoTokenizer.from_pretrained(model_name)
            print(f"[download] CausalLM: {model_name}")
            AutoModelForCausalLM.from_pretrained(model_name)

    print("[download] Completed.")



