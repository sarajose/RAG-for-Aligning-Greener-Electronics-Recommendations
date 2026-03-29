"""CLI parser for the full-study workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import OUTPUT_DIR

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
