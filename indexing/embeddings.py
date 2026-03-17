from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DEFAULT_MODEL_KEY, EMBEDDING_MODELS


def get_embed_model(model_key: str = DEFAULT_MODEL_KEY) -> SentenceTransformer:
    """Load a sentence-transformer model by configured key."""
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(EMBEDDING_MODELS)}")

    model_name = EMBEDDING_MODELS[model_key]
    print(f"[embed] Loading {model_name}")
    model = SentenceTransformer(model_name)
    setattr(model, "_model_key", model_key)
    return model


def _format_texts_for_model(
    texts: list[str],
    model: SentenceTransformer,
    *,
    is_query: bool,
) -> list[str]:
    """Apply model-specific prefixes when needed.

    E5 models require task prefixes: query: and passage:.
    """
    model_key = str(getattr(model, "_model_key", ""))
    if not model_key.startswith("e5-"):
        return texts

    prefix = "query: " if is_query else "passage: "
    return [f"{prefix}{text}" for text in texts]


def get_model_max_tokens(model: SentenceTransformer) -> int:
    """Get model max token length with safe fallback."""
    if hasattr(model, "max_seq_length") and model.max_seq_length:
        return int(model.max_seq_length)

    try:
        tokenizer: Any = model.tokenizer
        max_len = getattr(tokenizer, "model_max_length", 0)
        if max_len and max_len < 1_000_000:
            return int(max_len)
    except Exception:
        pass

    return 512


def check_token_lengths(
    texts: list[str],
    model: SentenceTransformer,
    *,
    warn: bool = True,
) -> dict[str, float]:
    """Analyze token lengths against model limits."""
    tokenizer = model.tokenizer
    limit = get_model_max_tokens(model)
    lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]
    over = [length for length in lengths if length > limit]

    summary = {
        "model_limit": float(limit),
        "max_tokens": float(max(lengths) if lengths else 0),
        "mean_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "n_over": float(len(over)),
        "pct_over": (100.0 * len(over) / len(lengths)) if lengths else 0.0,
        "longest": float(max(lengths) if lengths else 0),
    }

    if warn and over:
        print(
            f"[embed] WARNING: {len(over)}/{len(lengths)} texts "
            f"({summary['pct_over']:.1f}%) exceed the model limit {limit} "
            f"(longest: {int(summary['longest'])})."
        )
    elif warn:
        print(
            f"[embed] All {len(lengths)} texts fit in the model limit "
            f"{limit} (longest: {int(summary['longest'])})."
        )

    return summary


def _truncate_overlong_texts(texts: list[str], model: SentenceTransformer) -> tuple[list[str], int]:
    """Truncate only inputs that exceed the model token limit.

    This avoids rare OOM failures from extreme outlier chunks while keeping
    the rest of the corpus unchanged.
    """
    tokenizer = model.tokenizer
    limit = get_model_max_tokens(model)
    out: list[str] = []
    truncated = 0

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) <= limit:
            out.append(text)
            continue

        truncated += 1
        clipped = tokenizer.decode(
            token_ids[:limit],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        out.append(clipped)

    return out, truncated


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
    *,
    is_query: bool = False,
) -> np.ndarray:
    """Encode texts into normalized float32 embeddings."""
    formatted = _format_texts_for_model(texts, model, is_query=is_query)
    check_token_lengths(formatted, model, warn=show_progress)
    formatted, n_truncated = _truncate_overlong_texts(formatted, model)
    if show_progress and n_truncated:
        print(f"[embed] Truncated {n_truncated}/{len(formatted)} overlong texts to model token limit.")

    candidate_batch_sizes: list[int] = []
    for bs in (batch_size, 16, 8, 4, 2, 1):
        if bs > 0 and bs not in candidate_batch_sizes:
            candidate_batch_sizes.append(bs)

    last_error: Exception | None = None
    for bs in candidate_batch_sizes:
        try:
            if show_progress and bs != batch_size:
                print(f"[embed] Retrying with smaller batch_size={bs} due memory pressure.")
            return model.encode(
                formatted,
                batch_size=bs,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "not enough memory" in msg:
                last_error = e
                continue
            raise

    if last_error is not None:
        raise last_error

    raise RuntimeError("Embedding failed before model.encode could be executed.")
