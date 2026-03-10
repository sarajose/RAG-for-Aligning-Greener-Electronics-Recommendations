"""Evaluation package — retrieval and classification quality metrics."""

from evaluation.metrics import (
    compute_retrieval_metrics,
    compute_classification_metrics,
    bootstrap_ci,
    paired_permutation_test,
)
from evaluation.evaluation import (
    evaluate_retrieval,
    evaluate_paragraph_retrieval,
    per_query_retrieval_scores,
    load_gold_standard,
    load_whitepaper_recommendations,
    group_gold_by_query,
    format_retrieval_report,
    format_classification_report,
    save_metrics_json,
)

__all__ = [
    "compute_retrieval_metrics",
    "compute_classification_metrics",
    "bootstrap_ci",
    "paired_permutation_test",
    "evaluate_retrieval",
    "evaluate_paragraph_retrieval",
    "per_query_retrieval_scores",
    "load_gold_standard",
    "load_whitepaper_recommendations",
    "group_gold_by_query",
    "format_retrieval_report",
    "format_classification_report",
    "save_metrics_json",
]
