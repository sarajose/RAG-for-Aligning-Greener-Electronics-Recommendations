"""Compatibility facade for indexing utilities.

Public functions are re-exported from smaller modules under indexing/.
This keeps legacy imports working while reducing file size and complexity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_MODEL_KEY, EMBEDDING_MODELS
from indexing import (
    build_bm25_index,
    build_faiss_index,
    build_index,
    build_merged_index,
    check_token_lengths,
    embed_texts,
    get_embed_model,
    get_model_max_tokens,
    load_and_merge_chunks,
    load_chunks,
    load_indices,
    save_indices,
    tokenize,
)

__all__ = [
    "load_chunks",
    "load_and_merge_chunks",
    "get_embed_model",
    "get_model_max_tokens",
    "check_token_lengths",
    "embed_texts",
    "build_faiss_index",
    "build_bm25_index",
    "tokenize",
    "save_indices",
    "load_indices",
    "build_index",
    "build_merged_index",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build embedding + BM25 indices from evidence CSV files"
    )
    parser.add_argument("-i", "--input", required=True, type=Path, nargs="+")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_KEY, choices=list(EMBEDDING_MODELS))
    args = parser.parse_args()

    if len(args.input) == 1:
        build_index(args.input[0], args.model)
    else:
        build_merged_index(*args.input, model_key=args.model)


if __name__ == "__main__":
    main()
