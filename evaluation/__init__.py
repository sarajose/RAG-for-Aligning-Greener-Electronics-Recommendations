"""Evaluation package — retrieval and classification quality metrics."""

from evaluation.evaluation import (
    bootstrap_ci,
    compute_classification_metrics,
    compute_retrieval_metrics,
    evaluate_retrieval,
    group_gold_by_query,
    load_gold_standard,
    load_whitepaper_recommendations,
    paired_permutation_test,
    per_query_retrieval_scores,
)

__all__ = [
    "bootstrap_ci",
    "compute_classification_metrics",
    "compute_retrieval_metrics",
    "evaluate_retrieval",
    "group_gold_by_query",
    "load_gold_standard",
    "load_whitepaper_recommendations",
    "paired_permutation_test",
    "per_query_retrieval_scores",
]
