"""
Evaluation suite for the RAG policy-alignment pipeline.

Responsibilities
----------------
1. **Document-level retrieval evaluation** against the manually-annotated
   gold standard (``gold_standard_doc_level/gold_standard.csv``).
2. **External benchmark evaluation** (e.g. MLEB GDPR Holdings) for
   validating retrieval on out-of-domain legal data.
3. **Classification evaluation** against gold-standard alignment labels.
4. **Report generation** — human-readable tables and JSON export.

All heavy metric computation is delegated to :pymod:`metrics`.

Usage (standalone)::

    python evaluation.py --gold gold_standard_doc_level/gold_standard.csv \\
                         --model bge-m3 --top-k 10
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from config import (
    ALIGNMENT_LABELS,
    DEFAULT_MODEL_KEY,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP,
    EVAL_K_VALUES,
    GOLD_STANDARD_CSV,
    OUTPUT_DIR,
    normalise_doc_name,
)
from data_models import (
    Chunk,
    GoldStandardEntry,
    ClassificationResult,
    RetrievalMetrics,
    ClassificationMetrics,
)
from evaluation.metrics import compute_retrieval_metrics, compute_classification_metrics

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Gold-standard loading
# ═════════════════════════════════════════════════════════════════════════════


def load_gold_standard(
    csv_path: Path = GOLD_STANDARD_CSV,
) -> list[GoldStandardEntry]:
    """Load the manually-annotated gold standard from CSV.

    Parameters
    ----------
    csv_path : Path
        CSV with at least ``recommendation_text`` and ``doc_short_name``.

    Returns
    -------
    list[GoldStandardEntry]
    """
    entries: list[GoldStandardEntry] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entries.append(GoldStandardEntry(
                paper=row.get("Paper", ""),
                source_page=row.get("source_page", ""),
                source_line=row.get("source_line", ""),
                recommendation_text=row.get("recommendation_text", ""),
                source_snippet_original=row.get("source_snippet_original", ""),
                recommendation_or_statement=row.get(
                    "recommendation_or_statement", ""
                ),
                doc_short_name=row.get("doc_short_name", ""),
                doc_type=row.get("doc_type", ""),
                doc_ref_num=row.get("doc_ref_num", ""),
                doc_reference_raw_excerpt=row.get(
                    "doc_reference_raw_excerpt", ""
                ),
                evidence_span=row.get("evidence_span", ""),
                reference_basis=row.get("reference_basis", ""),
                needs_review=row.get("needs_review", ""),
                context_excerpt=row.get("context_excerpt", ""),
                alignment_label=row.get("alignment_label") or None,
            ))
    logger.info(
        "Loaded %d gold-standard entries from %s", len(entries), csv_path,
    )
    return entries


def group_gold_by_query(
    entries: list[GoldStandardEntry],
) -> dict[str, set[str]]:
    """Group gold-standard entries into ``{query: {canonical_doc_names}}``.

    Multiple entries with the same ``recommendation_text`` are merged so
    that a single query maps to the full set of relevant documents.

    Parameters
    ----------
    entries : list[GoldStandardEntry]

    Returns
    -------
    dict[str, set[str]]
    """
    grouped: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        canon = normalise_doc_name(e.doc_short_name)
        grouped[e.recommendation_text].add(canon)
    return dict(grouped)

# 2. Document-level retrieval evaluation (in-domain gold standard)

def evaluate_retrieval(
    retriever,  # HybridRetriever — imported lazily to avoid circular deps
    gold_path: Path = GOLD_STANDARD_CSV,
    k_values: Optional[list[int]] = None,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
) -> dict[int, RetrievalMetrics]:
    """Run document-level retrieval evaluation against the gold standard.

    For every unique recommendation in the gold standard the retriever
    fetches chunks.  The *documents* represented in those chunks are
    compared against the gold-standard relevant-document set.

    Parameters
    ----------
    retriever : HybridRetriever
        Fully initialised retriever (indices loaded).
    gold_path : Path
        Gold-standard CSV path.
    k_values : list[int] | None
        Cut-off depths to report (defaults to ``EVAL_K_VALUES``).
    top_k_retrieve : int
        Chunks to retrieve per query (before document-level dedup).
    rerank_top : int
        Chunks kept after cross-encoder reranking.

    Returns
    -------
    dict[int, RetrievalMetrics]
        ``{k: metrics}`` for each requested cut-off.
    """
    if k_values is None:
        k_values = EVAL_K_VALUES

    entries = load_gold_standard(gold_path)
    query_to_docs = group_gold_by_query(entries)
    queries = list(query_to_docs.keys())
    print(f"[eval] {len(queries)} unique queries from gold standard")

    # Ensure we retrieve enough chunks so that after document-level dedup
    # we still have at least max(k_values) distinct documents.
    max_k = max(k_values)
    n_retrieve = max(max_k * 3, top_k_retrieve, 30)
    n_rerank = max(max_k, rerank_top)

    all_retrieved_docs: list[list[str]] = []
    all_relevant_docs: list[set[str]] = []

    for i, query in enumerate(queries):
        result = retriever.retrieve(
            query, top_k=n_retrieve, rerank_top=n_rerank,
        )
        # Deduplicate to document-level ranking (preserve first-seen order)
        seen: set[str] = set()
        doc_ranking: list[str] = []
        for chunk in result.ranked_chunks:
            canon = normalise_doc_name(chunk.document)
            if canon not in seen:
                seen.add(canon)
                doc_ranking.append(canon)

        all_retrieved_docs.append(doc_ranking)
        all_relevant_docs.append(query_to_docs[query])

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(queries)}] queries processed")

    print(f"[eval] All {len(queries)} queries processed")

    # Compute metrics at each cut-off
    results: dict[int, RetrievalMetrics] = {}
    for k in sorted(k_values):
        results[k] = compute_retrieval_metrics(
            all_retrieved_docs, all_relevant_docs, k,
        )
    return results


# 3. External benchmark evaluation

def load_benchmark(benchmark_path: Path) -> list[dict]:
    """Load an external retrieval benchmark from a JSON file.

    Expected schema::

        [
          {
            "query": "text …",
            "relevant_ids": ["chunk_id_1", "chunk_id_2"]
          },
          …
        ]

    For document-level benchmarks, use ``"relevant_docs"`` instead and
    pass ``id_field="relevant_docs"`` to :func:`evaluate_benchmark`.

    Parameters
    ----------
    benchmark_path : Path
        JSON file with benchmark queries and gold labels.

    Returns
    -------
    list[dict]
    """
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        "Loaded %d benchmark queries from %s", len(data), benchmark_path,
    )
    return data


def evaluate_benchmark(
    retriever,
    benchmark_path: Path,
    k_values: Optional[list[int]] = None,
    top_k_retrieve: int = DEFAULT_TOP_K,
    rerank_top: int = DEFAULT_RERANK_TOP,
    id_field: str = "relevant_ids",
) -> dict[int, RetrievalMetrics]:
    """Run retrieval evaluation on an external benchmark dataset.

    Works with **chunk-level** (``relevant_ids``) or **document-level**
    (``relevant_docs``) gold annotations depending on *id_field*.

    Parameters
    ----------
    retriever : HybridRetriever
    benchmark_path : Path
    k_values : list[int] | None
    top_k_retrieve : int
    rerank_top : int
    id_field : str
        JSON key holding the relevant identifiers per query.

    Returns
    -------
    dict[int, RetrievalMetrics]
    """
    if k_values is None:
        k_values = EVAL_K_VALUES

    cases = load_benchmark(benchmark_path)
    print(f"[benchmark] {len(cases)} queries loaded")

    max_k = max(k_values)
    n_retrieve = max(max_k * 3, top_k_retrieve, 30)
    n_rerank = max(max_k, rerank_top)

    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []

    for i, case in enumerate(cases):
        query = case["query"]
        relevant = set(case.get(id_field, []))

        result = retriever.retrieve(
            query, top_k=n_retrieve, rerank_top=n_rerank,
        )
        retrieved_ids = [c.id for c in result.ranked_chunks]

        all_retrieved.append(retrieved_ids)
        all_relevant.append(relevant)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(cases)}] benchmark queries processed")

    results: dict[int, RetrievalMetrics] = {}
    for k in sorted(k_values):
        results[k] = compute_retrieval_metrics(all_retrieved, all_relevant, k)
    return results


# 4. Classification evaluation

def evaluate_classification(
    predictions: list[ClassificationResult],
    gold_labels: list[str],
    labels: Optional[list[str]] = None,
) -> ClassificationMetrics:
    """Evaluate LLM alignment classification against gold labels.

    Parameters
    ----------
    predictions : list[ClassificationResult]
        LLM outputs (same order as *gold_labels*).
    gold_labels : list[str]
        Ground-truth alignment labels.
    labels : list[str] | None
        Label vocabulary (defaults to ``ALIGNMENT_LABELS``).

    Returns
    -------
    ClassificationMetrics
    """
    if labels is None:
        labels = ALIGNMENT_LABELS
    y_pred = [p.label for p in predictions]
    return compute_classification_metrics(gold_labels, y_pred, labels)


# 5. Report formatting

def format_retrieval_report(
    metrics_by_k: dict[int, RetrievalMetrics],
    title: str = "Retrieval Evaluation",
) -> str:
    """Format retrieval metrics into a human-readable table.

    Parameters
    ----------
    metrics_by_k : dict[int, RetrievalMetrics]
    title : str

    Returns
    -------
    str
        Multi-line report ready for ``print()``.
    """
    m0 = next(iter(metrics_by_k.values()))
    lines = [
        "",
        "=" * 72,
        f"  {title}  (n = {m0.num_queries} queries)",
        "=" * 72,
        f"{'k':>4} | {'Hit@k':>7} | {'Recall':>7} | {'Prec':>7} "
        f"| {'MRR':>7} | {'MAP':>7} | {'NDCG':>7}",
        "-" * 72,
    ]
    for k, m in sorted(metrics_by_k.items()):
        lines.append(
            f"{k:>4} | {m.hit_rate:>7.3f} | {m.recall:>7.3f} | "
            f"{m.precision:>7.3f} | {m.mrr:>7.3f} | "
            f"{m.map_score:>7.3f} | {m.ndcg:>7.3f}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def format_classification_report(
    metrics: ClassificationMetrics,
    title: str = "Classification Evaluation",
) -> str:
    """Format classification metrics into a human-readable report.

    Parameters
    ----------
    metrics : ClassificationMetrics
    title : str

    Returns
    -------
    str
        Multi-line report ready for ``print()``.
    """
    lines = [
        "",
        "=" * 72,
        f"  {title}  (n = {metrics.num_samples} samples)",
        "=" * 72,
        f"  Accuracy:        {metrics.accuracy:.3f}",
        f"  Macro-F1:        {metrics.macro_f1:.3f}",
        f"  Weighted-F1:     {metrics.weighted_f1:.3f}",
        f"  Cohen's Kappa:   {metrics.cohens_kappa:.3f}",
        "",
        f"  {'Label':<28} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}",
        f"  {'-' * 55}",
    ]
    for lbl in metrics.labels:
        s = metrics.per_class.get(lbl, {})
        lines.append(
            f"  {lbl:<28} | {s.get('precision', 0):>6.3f} | "
            f"{s.get('recall', 0):>6.3f} | {s.get('f1', 0):>6.3f}"
        )
    lines.append("")
    lines.append("  Confusion matrix (rows = true, cols = predicted):")
    lines.append(f"  Labels: {metrics.labels}")
    for row in metrics.confusion_matrix:
        lines.append(f"    {row}")
    lines.append("=" * 72)
    return "\n".join(lines)


# 6. Persistence

def save_metrics_json(
    retrieval_metrics: Optional[dict[int, RetrievalMetrics]] = None,
    classification_metrics: Optional[ClassificationMetrics] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Save all metrics to a JSON file for downstream analysis.

    Parameters
    ----------
    retrieval_metrics : dict[int, RetrievalMetrics] | None
    classification_metrics : ClassificationMetrics | None
    output_path : Path | None
        Destination JSON file (defaults to ``outputs/metrics.json``).
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "metrics.json"

    payload: dict = {}
    if retrieval_metrics:
        payload["retrieval"] = {
            str(k): asdict(m) for k, m in retrieval_metrics.items()
        }
    if classification_metrics:
        payload["classification"] = asdict(classification_metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[eval] Metrics saved → {output_path}")


def save_per_query_results(
    queries: list[str],
    all_retrieved_docs: list[list[str]],
    all_relevant_docs: list[set[str]],
    output_path: Path,
) -> None:
    """Write per-query retrieval breakdown for error analysis.

    Parameters
    ----------
    queries : list[str]
    all_retrieved_docs : list[list[str]]
    all_relevant_docs : list[set[str]]
    output_path : Path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for query, retrieved, relevant in zip(
        queries, all_retrieved_docs, all_relevant_docs,
    ):
        hit = bool(relevant & set(retrieved[:10]))
        rows.append({
            "query": query,
            "relevant_docs": "; ".join(sorted(relevant)),
            "retrieved_docs_top10": "; ".join(retrieved[:10]),
            "hit_at_10": int(hit),
            "recall_at_10": (
                len(relevant & set(retrieved[:10])) / len(relevant)
                if relevant else 0.0
            ),
        })
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Per-query results saved → {output_path}")


# 7. CLI

def main() -> None:
    """Run evaluation from the command line."""
    import argparse
    from retrieval.retrieval import HybridRetriever  # deferred to avoid circular import

    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and classification quality",
    )
    parser.add_argument(
        "--gold", type=Path, default=GOLD_STANDARD_CSV,
        help="Gold-standard CSV path",
    )
    parser.add_argument(
        "--benchmark", type=Path, default=None,
        help="External benchmark JSON path (optional)",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        help="Embedding model key",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=DEFAULT_TOP_K,
    )
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="JSON file for metric export",
    )
    parser.add_argument(
        "--per-query", type=Path, default=None,
        help="CSV path for per-query retrieval breakdown",
    )
    args = parser.parse_args()

    retriever = HybridRetriever.from_disk(
        args.model, use_reranker=not args.no_rerank,
    )

    # ── In-domain (gold standard) ──
    ret_metrics = None
    if args.gold and args.gold.exists():
        print("\n>>> Document-level retrieval evaluation (gold standard)")
        ret_metrics = evaluate_retrieval(
            retriever,
            args.gold,
            top_k_retrieve=args.top_k,
            rerank_top=args.rerank_top,
        )
        print(format_retrieval_report(ret_metrics, "Gold-Standard Retrieval"))

    # ── External benchmark ──
    bench_metrics = None
    if args.benchmark and args.benchmark.exists():
        print("\n>>> External benchmark evaluation")
        bench_metrics = evaluate_benchmark(
            retriever,
            args.benchmark,
            top_k_retrieve=args.top_k,
            rerank_top=args.rerank_top,
        )
        print(format_retrieval_report(bench_metrics, "External Benchmark"))

    # ── Save ──
    if args.output:
        save_metrics_json(ret_metrics, None, args.output)


if __name__ == "__main__":
    main() 