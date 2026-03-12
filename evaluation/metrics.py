"""
Retrieval and classification metric computation.

All functions are **stateless**: they receive predictions and ground-truth
labels and return numeric scores.  The evaluation *orchestration* (loading
gold standards, running the retriever, aggregating results) lives in
:pymod:`evaluation`.

Retrieval metrics
-----------------
* Hit@k, Recall@k, Precision@k
* MRR (Mean Reciprocal Rank)
* MAP (Mean Average Precision)
* NDCG (Normalised Discounted Cumulative Gain)

Classification metrics
----------------------
* Accuracy, Macro-F1, Weighted-F1
* Cohen's Kappa
* Per-class Precision / Recall / F1
* Confusion matrix
"""


import math
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix as sk_confusion_matrix,
    cohen_kappa_score,
)

from data_models import RetrievalMetrics, ClassificationMetrics


# Bootstrap confidence intervals & statistical significance

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for the mean of *scores*.

    Parameters
    ----------
    scores : list[float]
        Per-query metric values (e.g. per-query NDCG).
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95 → 95 % CI).
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, ci_lower, ci_upper)``
    """
    rng = np.random.RandomState(rng_seed)
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = arr[rng.randint(0, n, size=n)]
        means[i] = sample.mean()
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def paired_permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10_000,
    rng_seed: int = 42,
) -> float:
    """Two-sided paired permutation test for the difference of means.

    Tests whether system A and system B perform differently on the same
    set of queries.

    Parameters
    ----------
    scores_a, scores_b : list[float]
        Per-query scores for system A and system B (same length / order).
    n_permutations : int
        Number of random permutations.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Two-sided p-value.
    """
    rng = np.random.RandomState(rng_seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert len(a) == len(b), "Score lists must have the same length"
    diff = a - b
    observed = np.abs(diff.mean())
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        if np.abs((diff * signs).mean()) >= observed:
            count += 1
    return count / n_permutations

# Retrieval — per-query helpers

def hit_at_k(retrieved: list[str], relevant: set[str]) -> int:
    """Return 1 if **any** relevant item appears in *retrieved*, else 0.

    Parameters
    ----------
    retrieved : list[str]
        Ordered list of retrieved identifiers (already cut to depth *k*
        by the caller).
    relevant : set[str]
        Ground-truth relevant identifiers.
    """
    return int(bool(relevant & set(retrieved)))


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Fraction of relevant items found in *retrieved*.

    Returns 0.0 when *relevant* is empty.
    """
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved)) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-*k* that are relevant."""
    if k == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0 if none)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def rank_of_first_relevant(retrieved: list[str], relevant: set[str]) -> float:
    """Return the 1-based rank of the first relevant item.

    Returns ``float('inf')`` when no relevant item is found in
    *retrieved*.  Useful for computing Mean Rank (MR) when each
    query has exactly one relevant document.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return float(rank)
    return float("inf")

def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average precision for a single query (binary relevance)."""
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(relevant)

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at cut-off *k*.

    Uses **binary** relevance (1 if relevant, 0 otherwise).
    """
    def _dcg(rels: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    rels = [1 if d in relevant else 0 for d in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0



# Retrieval - aggregated over query set

def compute_retrieval_metrics(
    all_retrieved: list[list[str]],
    all_relevant: list[set[str]],
    k: int,
) -> RetrievalMetrics:
    """Aggregate retrieval metrics over a set of queries.

    Parameters
    ----------
    all_retrieved : list[list[str]]
        For each query, an ordered list of retrieved identifiers.
    all_relevant : list[set[str]]
        For each query, the set of ground-truth relevant identifiers.
    k : int
        Cut-off depth.

    Returns
    -------
    RetrievalMetrics
    """
    n = len(all_retrieved)
    if n != len(all_relevant):
        raise ValueError(
            f"Length mismatch: {n} retrieved lists vs "
            f"{len(all_relevant)} relevant sets"
        )

    hits = [hit_at_k(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    recalls = [recall_at_k(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    precs = [precision_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    mrrs = [reciprocal_rank(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    aps = [average_precision(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]
    ndcgs = [ndcg_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    ranks = [rank_of_first_relevant(r[:k], rel) for r, rel in zip(all_retrieved, all_relevant)]

    # Mean rank: only average over queries where the item was actually found
    found_ranks = [r for r in ranks if r != float("inf")]
    mean_r = float(np.mean(found_ranks)) if found_ranks else float("inf")

    return RetrievalMetrics(
        k=k,
        hit_rate=float(np.mean(hits)),
        recall=float(np.mean(recalls)),
        precision=float(np.mean(precs)),
        mrr=float(np.mean(mrrs)),
        map_score=float(np.mean(aps)),
        ndcg=float(np.mean(ndcgs)),
        num_queries=n,
        mean_rank=mean_r,
    )


# Classification metrics
def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> ClassificationMetrics:
    """Compute accuracy, F1 variants, kappa, and confusion matrix.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth alignment labels.
    y_pred : list[str]
        Predicted alignment labels.
    labels : list[str]
        Ordered list of all possible labels.

    Returns
    -------
    ClassificationMetrics
    """
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )
    weighted = f1_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0,
    )
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = {
        lbl: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        for lbl, p, r, f in zip(labels, prec_arr, rec_arr, f1_arr)
    }

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return ClassificationMetrics(
        accuracy=float(acc),
        macro_f1=float(macro),
        weighted_f1=float(weighted),
        cohens_kappa=float(kappa),
        per_class=per_class,
        confusion_matrix=cm,
        labels=labels,
        num_samples=len(y_true),
    )
