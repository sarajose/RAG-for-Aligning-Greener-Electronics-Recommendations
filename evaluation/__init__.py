"""Evaluation package — retrieval and classification quality metrics."""

from evaluation.metrics import (
    bootstrap_ci,
    compute_classification_metrics,
    compute_retrieval_metrics,
    paired_permutation_test,
)
from evaluation.evaluation import (
    evaluate_retrieval,
    format_retrieval_report,
    group_gold_by_query,
    load_gold_standard,
    load_whitepaper_recommendations,
    per_query_retrieval_scores,
)

__all__ = [
    "compute_retrieval_metrics",
    "compute_classification_metrics",
    "bootstrap_ci",
    "paired_permutation_test",
    "evaluate_retrieval",
    "format_retrieval_report",
    "group_gold_by_query",
    "load_gold_standard",
    "load_whitepaper_recommendations",
    "per_query_retrieval_scores",
]
