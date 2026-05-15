"""Prompt study analysis helper.

Analyzes classification and judge outputs from `main.py prompt --judge`,
generating distribution CSVs that are read by the classifier and judge notebooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Allow running as `python evaluation/full_study.py ...` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thesis evaluation orchestration")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompt = sub.add_parser("prompt-study", help="Analyze prompt classification and judge outputs")
    p_prompt.add_argument("--prompt-csv", type=Path, required=True)
    p_prompt.add_argument("--judge-csv", type=Path, default=None)
    p_prompt.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "eval_prompt")

    return parser


def _split_semicolon_values(series: pd.Series) -> pd.Series:
    """Flatten a Series of semicolon-separated strings into individual values."""
    cleaned = series.fillna("").astype(str).str.strip()
    values = cleaned.str.split(";")
    flattened = [item.strip() for sub in values for item in sub if item and item.strip()]
    return pd.Series(flattened, dtype="string")


def run_prompt_study(args: argparse.Namespace) -> None:
    """Analyze classification and judge CSVs; write distribution CSVs to output_dir."""
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
        report["judge_summary"] = {
            "num_rows": int(len(judge_df)),
            "label_score_mean": float(judge_df["label_score"].mean()),
            "justification_score_mean": float(judge_df["justification_score"].mean()),
            "evidence_score_mean": float(judge_df["evidence_score"].mean()),
            "overall_score_mean": float(judge_df["overall_score"].mean()),
            "overall_score_std": float(judge_df["overall_score"].std(ddof=0)),
        }

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
    if args.command == "prompt-study":
        run_prompt_study(args)


if __name__ == "__main__":
    main()
