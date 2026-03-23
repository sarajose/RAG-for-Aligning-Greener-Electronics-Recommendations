"""CLI pipeline for the thesis workflow.

Commands:
  build           — embed and index evidence documents
  prompt          — retrieve + classify recommendations (+ optional LLM judge)
  evaluate        — unified retrieval evaluation with ablation study
  download-models — pre-cache embedding/reranker/LLM models
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import (
    DEFAULT_MODEL_KEY,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RERANK_TOP,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    EVAL_K_VALUES,
    GOLD_STANDARD_CSV,
    OUTPUT_DIR,
    RETRIEVAL_MODES,
    WHITEPAPER_RECOMMENDATIONS_CSV,
)
from pipeline_commands import cmd_build, cmd_download_models, cmd_evaluate, cmd_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Policy-Alignment Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ──────────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Embed and index evidence CSV files")
    p_build.add_argument("-i", "--input", required=True, type=Path, nargs="+", help="Evidence CSV file(s)")
    p_build.add_argument("-m", "--model", default=DEFAULT_MODEL_KEY, choices=list(EMBEDDING_MODELS))

    # ── prompt ─────────────────────────────────────────────────────────────
    p_prompt = sub.add_parser("prompt", help="Retrieve, classify, and optionally judge recommendations")
    p_prompt.add_argument(
        "-i", "--input", type=Path, default=WHITEPAPER_RECOMMENDATIONS_CSV,
        help="Recommendations CSV (semicolon or comma delimited)",
    )
    p_prompt.add_argument(
        "-o", "--output", type=Path, default=OUTPUT_DIR / "prompt_results.csv",
        help="Output CSV for results",
    )
    p_prompt.add_argument("-m", "--model", default=DEFAULT_MODEL_KEY, choices=list(EMBEDDING_MODELS))
    p_prompt.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_prompt.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    p_prompt.add_argument(
        "--retrieval-mode", default=DEFAULT_RETRIEVAL_MODE, choices=RETRIEVAL_MODES,
        help="flat_baseline (default) or split_evidence_retrieval",
    )
    p_prompt.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    p_prompt.add_argument("--retrieve-only", action="store_true", help="Skip LLM classification")
    p_prompt.add_argument("--judge", action="store_true", help="Run LLM judge after classification")

    # ── evaluate ───────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Run unified retrieval evaluation")
    p_eval.add_argument(
        "--models", nargs="+", default=[DEFAULT_MODEL_KEY, "e5-large-v2", "e5-mistral"],
        help="Embedding model keys to compare (must have pre-built indices)",
    )
    p_eval.add_argument("--gold-csv", type=Path, default=GOLD_STANDARD_CSV)
    p_eval.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "eval_unified")
    p_eval.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_eval.add_argument("--k-values", type=int, nargs="+", default=EVAL_K_VALUES)
    p_eval.add_argument("--skip-mteb", action="store_true", help="Skip MTEB legal tasks (faster)")
    p_eval.add_argument("--skip-reranker", action="store_true", help="Skip cross-encoder reranking")
    p_eval.add_argument("--force-cpu", action="store_true", help="Disable GPU")
    p_eval.add_argument(
        "--with-robustness", action="store_true",
        help="Also run ablation significance tests (paired permutation + bootstrap CI)",
    )

    # ── download-models ────────────────────────────────────────────────────
    p_dl = sub.add_parser("download-models", help="Pre-download embedding/reranker/LLM models")
    p_dl.add_argument(
        "--embedding-models", nargs="+", default=["bge-m3", "e5-large-v2", "e5-mistral"],
    )
    p_dl.add_argument("--include-llms", action="store_true")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "prompt":
        cmd_prompt(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "download-models":
        cmd_download_models(args)


if __name__ == "__main__":
    main()
