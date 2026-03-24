"""Standalone entrypoint for unified evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_RERANK_TOP,
    DEFAULT_TOP_K,
    EVAL_K_VALUES,
    EVIDENCE_CSV,
    GOLD_STANDARD_CSV,
    OUTPUT_DIR,
    SPLADE_MAX_LENGTH,
    SPLADE_MODEL,
    WHITEPAPER_RECOMMENDATIONS_CSV,
)
from evaluation.experiment_commands import cmd_robustness, cmd_unified_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone unified evaluation CLI")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL_KEY])
    parser.add_argument("--gold-csv", type=Path, default=GOLD_STANDARD_CSV)
    parser.add_argument("--whitepaper-csv", type=Path, default=WHITEPAPER_RECOMMENDATIONS_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "eval_unified")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    parser.add_argument("--export-k", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=EVAL_K_VALUES)
    parser.add_argument("--mteb-dataset", default="mteb/legalbench_consumer_contracts_qa")
    parser.add_argument("--mteb-split", default="test")
    parser.add_argument("--max-corpus", type=int, default=20000)
    parser.add_argument("--full-mteb", action="store_true")
    parser.add_argument("--skip-mteb", action="store_true")
    parser.add_argument("--skip-whitepaper", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--auto-build-indices", action="store_true")
    parser.add_argument("--evidence-csv", type=Path, default=EVIDENCE_CSV)
    parser.add_argument("--include-splade", action="store_true")
    parser.add_argument("--splade-model", default=SPLADE_MODEL)
    parser.add_argument("--splade-max-length", type=int, default=SPLADE_MAX_LENGTH)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--with-robustness", action="store_true")
    parser.add_argument("--robust-model", default=None)
    parser.add_argument("--robust-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd_unified_eval(args)

    if args.with_robustness:
        robust_args = argparse.Namespace(
            model=args.robust_model or args.models[0],
            gold_csv=args.gold_csv,
            k=args.robust_k,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            output_dir=args.output_dir / "robustness",
            skip_reranker=args.skip_reranker,
        )
        cmd_robustness(robust_args)


if __name__ == "__main__":
    main()
