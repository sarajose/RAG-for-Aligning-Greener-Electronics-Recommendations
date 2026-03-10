"""
Main CLI pipeline for the RAG policy-alignment system.

Orchestrates the full workflow: index building, retrieval, alignment
classification, and evaluation.

Sub-commands
------------
``build``
    Build embedding + BM25 indices from one or more evidence CSVs.
``evaluate``
    Evaluate retrieval quality on the gold standard and/or an external
    benchmark.
``classify``
    Run RAG alignment classification on a set of recommendations.
``run``
    Full pipeline: retrieve → classify → evaluate (gold standard available).

Examples::

    # Build indices (can merge multiple CSVs)
    python pipeline.py build -i outputs/evidence.csv outputs/evidence_recommendation.csv -m bge-m3

    # Evaluate retrieval on gold standard
    python pipeline.py evaluate --gold gold_standard_doc_level/gold_standard.csv

    # Classify recommendations
    python pipeline.py classify -i outputs/recommendations.csv -o outputs/classified.csv

    # Full pipeline
    python pipeline.py run -i outputs/recommendations.csv \\
                           --gold gold_standard_doc_level/gold_standard.csv \\
                           -o outputs/classified.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

from config import (
    ALIGNMENT_LABELS,
    DEFAULT_MODEL_KEY,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP,
    EMBEDDING_MODELS,
    EVAL_K_VALUES,
    GOLD_STANDARD_CSV,
    OUTPUT_DIR,
)
from data_models import ClassificationResult, Recommendation
from embedding_indexing import build_index, build_merged_index
from retrieval.retrieval import HybridRetriever
from evaluation.evaluation import (
    evaluate_retrieval,
    evaluate_benchmark,
    evaluate_classification,
    format_retrieval_report,
    format_classification_report,
    save_metrics_json,
    save_per_query_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Helpers
def load_recommendations(csv_path: Path) -> list[Recommendation]:
    """Load recommendations from a CSV file.

    Expected columns: ``section, subsection, title, recommendation``.

    Parameters
    ----------
    csv_path : Path
        CSV produced by :pymod:`chunking_recommendations`.

    Returns
    -------
    list[Recommendation]
    """
    recs: list[Recommendation] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            recs.append(Recommendation(
                section=row.get("section", ""),
                subsection=row.get("subsection", ""),
                title=row.get("title", ""),
                text=row.get("recommendation", row.get("text", "")),
            ))
    return recs


def _save_classification_csv(
    results: list[ClassificationResult],
    recs: list[Recommendation],
    output_path: Path,
) -> None:
    """Write classification results to a CSV file.

    Each row contains the recommendation context, predicted label,
    justification, cited chunk IDs, and the evidence documents/articles
    seen by the LLM.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for res in results:
        # Locate the matching recommendation for section/title context
        matching = next(
            (r for r in recs if r.text == res.recommendation), None,
        )
        rows.append({
            "section": matching.section if matching else "",
            "subsection": matching.subsection if matching else "",
            "title": matching.title if matching else "",
            "recommendation": res.recommendation,
            "alignment_label": res.label,
            "justification": res.justification,
            "cited_chunk_ids": "; ".join(res.cited_chunk_ids),
            "evidence_documents": "; ".join(sorted(
                {c.document for c in res.retrieved_chunks}
            )),
            "evidence_articles": "; ".join(sorted(
                {c.article for c in res.retrieved_chunks if c.article}
            )),
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[classify] Results saved → {output_path}")


# Sub-commands
def cmd_build(args: argparse.Namespace) -> None:
    """Build embedding + BM25 indices from evidence CSVs."""
    if len(args.input) == 1:
        build_index(args.input[0], args.model)
    else:
        build_merged_index(*args.input, model_key=args.model)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate retrieval on gold standard and/or an external benchmark."""
    retriever = HybridRetriever.from_disk(
        args.model, use_reranker=not args.no_rerank,
    )

    all_metrics: dict[str, dict[int, object]] = {}

    # ── Gold-standard retrieval ──
    if args.gold and args.gold.exists():
        print("  Gold-Standard Document-Level Retrieval Evaluation")
        ret_metrics = evaluate_retrieval(
            retriever,
            args.gold,
            top_k_retrieve=args.top_k,
            rerank_top=args.rerank_top,
        )
        all_metrics["gold_standard"] = ret_metrics
        print(format_retrieval_report(
            ret_metrics, "Gold-Standard Doc-Level Retrieval",
        ))

    # ── External benchmark ──
    if args.benchmark and args.benchmark.exists():

        print("  External Benchmark Retrieval Evaluation")
        bench_metrics = evaluate_benchmark(
            retriever,
            args.benchmark,
            top_k_retrieve=args.top_k,
            rerank_top=args.rerank_top,
        )
        all_metrics["benchmark"] = bench_metrics
        print(format_retrieval_report(
            bench_metrics, "External Benchmark Retrieval",
        ))

    # ── Save ──
    if args.output:
        # Save the first available metric set
        first_ret = next(iter(all_metrics.values()), None)
        save_metrics_json(first_ret, None, args.output)


def cmd_classify(args: argparse.Namespace) -> None:
    """Run RAG alignment classification on recommendations."""
    from rag.classifier import AlignmentClassifier

    retriever = HybridRetriever.from_disk(args.model)
    classifier = AlignmentClassifier()

    recs = load_recommendations(args.input)
    print(f"\n[classify] {len(recs)} recommendations loaded")

    results: list[ClassificationResult] = []
    for i, rec in enumerate(recs, 1):
        if not rec.text.strip():
            continue
        print(f"  [{i}/{len(recs)}] {rec.text[:80]}…")

        retrieval = retriever.retrieve(
            rec.text, top_k=args.top_k, rerank_top=args.rerank_top,
        )
        classification = classifier.classify(
            rec.text, retrieval.ranked_chunks,
        )
        results.append(classification)
        print(f"    → {classification.label}")

    if args.output:
        _save_classification_csv(results, recs, args.output)

    print(f"\n[classify] Done — {len(results)} recommendations classified.")


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: evaluate retrieval → classify → evaluate labels."""
    retriever = HybridRetriever.from_disk(args.model)

    # ── Step 1: Retrieval evaluation ──
    ret_metrics = None
    if args.gold and args.gold.exists():
        print("Step 1: Retrieval Evaluation (Gold Standard)")
        ret_metrics = evaluate_retrieval(
            retriever,
            args.gold,
            top_k_retrieve=args.top_k,
            rerank_top=args.rerank_top,
        )
        print(format_retrieval_report(
            ret_metrics, "Gold-Standard Doc-Level Retrieval",
        ))

    #RAG alignment classification
    results: list[ClassificationResult] = []
    recs: list[Recommendation] = []
    if args.input and args.input.exists():
        print("Step 2: RAG Alignment Classification")
        try:
            from rag.classifier import AlignmentClassifier
            classifier = AlignmentClassifier()
        except (EnvironmentError, ImportError) as exc:
            print(
                "[warn] OPENAI_API_KEY not set — skipping classification. "
                "Set it and re-run to enable."
            )
            classifier = None

        if classifier:
            recs = load_recommendations(args.input)
            for i, rec in enumerate(recs, 1):
                if not rec.text.strip():
                    continue
                print(f"  [{i}/{len(recs)}] {rec.text[:80]}…")
                retrieval = retriever.retrieve(
                    rec.text,
                    top_k=args.top_k,
                    rerank_top=args.rerank_top,
                )
                classification = classifier.classify(
                    rec.text, retrieval.ranked_chunks,
                )
                results.append(classification)
                print(f"    → {classification.label}")

            if args.output:
                _save_classification_csv(results, recs, args.output)

    # Classification evaluation (if gold labels exist)
    cls_metrics = None
    if results and args.gold and args.gold.exists():
        # Check whether the gold standard has alignment_label column
        from evaluation.evaluation import load_gold_standard
        gs_entries = load_gold_standard(args.gold)
        labeled = [e for e in gs_entries if e.alignment_label]
        if labeled:
            print("Step 3: Classification Evaluation")
            # Match predictions to gold labels by recommendation text
            gold_map = {e.recommendation_text: e.alignment_label for e in labeled}
            matched_preds = [
                r for r in results if r.recommendation in gold_map
            ]
            matched_gold = [
                gold_map[r.recommendation] for r in matched_preds
            ]
            if matched_preds:
                cls_metrics = evaluate_classification(
                    matched_preds, matched_gold,
                )
                print(format_classification_report(
                    cls_metrics, "Alignment Classification",
                ))
            else:
                print(
                    "[info] No matching gold labels found for these "
                    "recommendations."
                )

    # Save metrics
    metrics_path = (
        args.output.parent / "metrics.json"
        if args.output
        else OUTPUT_DIR / "metrics.json"
    )
    save_metrics_json(ret_metrics, cls_metrics, metrics_path)
    print("Pipeline complete")


# CLI
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Policy-Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser(
        "build", help="Build embedding + BM25 indices from evidence CSVs",
    )
    p_build.add_argument(
        "-i", "--input", required=True, type=Path, nargs="+",
        help="One or more evidence CSV files",
    )
    p_build.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )

    # evaluate
    p_eval = sub.add_parser(
        "evaluate", help="Evaluate retrieval on gold standard / benchmark",
    )
    p_eval.add_argument(
        "--gold", type=Path, default=GOLD_STANDARD_CSV,
        help="Gold-standard CSV path",
    )
    p_eval.add_argument(
        "--benchmark", type=Path, default=None,
        help="External benchmark JSON path (optional)",
    )
    p_eval.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )
    p_eval.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_eval.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    p_eval.add_argument("--no-rerank", action="store_true")
    p_eval.add_argument(
        "-o", "--output", type=Path, default=None,
        help="JSON path for metrics export",
    )

    # classify
    p_cls = sub.add_parser(
        "classify", help="Run RAG alignment classification",
    )
    p_cls.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Recommendations CSV",
    )
    p_cls.add_argument(
        "-o", "--output", type=Path,
        default=OUTPUT_DIR / "classified.csv",
        help="Output CSV for classified recommendations",
    )
    p_cls.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )
    p_cls.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_cls.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    p_cls.add_argument("--no-rerank", action="store_true")

    # run (full pipeline)
    p_run = sub.add_parser(
        "run", help="Full pipeline: retrieve → classify → evaluate",
    )
    p_run.add_argument(
        "-i", "--input", type=Path, default=None,
        help="Recommendations CSV (optional; skip classification if omitted)",
    )
    p_run.add_argument(
        "--gold", type=Path, default=GOLD_STANDARD_CSV,
        help="Gold-standard CSV path",
    )
    p_run.add_argument(
        "-o", "--output", type=Path,
        default=OUTPUT_DIR / "classified.csv",
        help="Output CSV for classified recommendations",
    )
    p_run.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )
    p_run.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_run.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    p_run.add_argument("--no-rerank", action="store_true")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
