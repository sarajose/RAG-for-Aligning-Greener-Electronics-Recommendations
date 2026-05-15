"""CLI command handlers for evaluation workflows."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from config import EMBEDDING_MODELS, JUDGE_MODEL, LLM_MODEL, RERANKER_MODEL
from evaluation.retrieval_eval import cmd_unified_eval

__all__ = ["cmd_unified_eval", "cmd_merge_eval", "cmd_download_models"]

def cmd_merge_eval(args: argparse.Namespace) -> None:
    """Merge one or more remote metrics CSVs into the local metrics_all.csv."""
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

    print("\n[done] Merged evaluation metrics.")
    print(f"[done] Metrics: {local_metrics_csv}")

def cmd_download_models(args: argparse.Namespace) -> None:
    """Pre-download embedding, reranker, and LLM models."""
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
