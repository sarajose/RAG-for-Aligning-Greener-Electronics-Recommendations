"""Evaluation package — retrieval and classification quality metrics."""

from evaluation.evaluation import (
    compute_retrieval_metrics,
    evaluate_retrieval,
    load_gold_standard,
)

__all__ = [
    "compute_retrieval_metrics",
    "evaluate_retrieval",
    "load_gold_standard",
]
