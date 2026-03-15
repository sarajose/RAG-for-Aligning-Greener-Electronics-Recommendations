"""
Main CLI pipeline for the RAG policy-alignment system.

Orchestrates the full workflow: index building, retrieval, alignment
classification, and evaluation.

Sub-commands
------------
``build``
    Build embedding + BM25 indices from one or more evidence CSVs.
``evaluate``
    Evaluate retrieval quality on the gold standard and/or MTEB benchmark
    data exported to JSON.
``classify``
    Run RAG alignment classification on a set of recommendations.
``whitepaper``
    Retrieve + classify the whitepaper recommendations (unlabelled) and
    export results for human evaluation.
``benchmark``
    Run retrieval evaluation on MTEB benchmark data (JSON).
``run``
    Full pipeline: retrieve → classify → evaluate (gold standard available).

Examples::

    # Build indices (can merge multiple CSVs)
    python pipeline.py build -i outputs/evidence.csv outputs/evidence_recommendation.csv -m bge-m3

    # Evaluate retrieval on gold standard
    python pipeline.py evaluate --gold gold_standard_doc_level/gold_standard.csv

    # Classify recommendations
    python pipeline.py classify -i outputs/recommendations.csv -o outputs/classified.csv

    # Run on the whitepaper recommendations
    python pipeline.py whitepaper -o outputs/whitepaper_classified.csv

    # Run an MTEB benchmark (JSON export)
    python pipeline.py benchmark -i benchmarks/my_benchmark.json -o outputs/benchmark_metrics.json

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
    BENCHMARK_DIR,
    DEFAULT_MODEL_KEY,
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP,
    EMBEDDING_MODELS,
    EVAL_K_VALUES,
    GOLD_STANDARD_CSV,
    OUTPUT_DIR,
    WHITEPAPER_RECOMMENDATIONS_CSV,
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


def _save_retrieved_chunks_csv(
    queries: list[str],
    retrieval_results: list,
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Save per-query retrieved chunks with full text to a CSV file.

    One row per (query, chunk) pair so every retrieved passage is inspectable
    for human evaluation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for query, result in zip(queries, retrieval_results):
        for rank, (chunk, score) in enumerate(
            zip(result.ranked_chunks[:top_k], result.scores[:top_k]), 1
        ):
            rows.append({
                "query": query,
                "rank": rank,
                "chunk_id": chunk.id,
                "document": chunk.document,
                "article": chunk.article,
                "paragraph": chunk.paragraph,
                "score": round(score, 6),
                "text": chunk.text,
            })
    if not rows:
        print("[pipeline] No retrieved chunks to save.")
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[pipeline] Retrieved chunks saved → {output_path}")


def _save_judge_results_csv(
    judge_results: list,
    output_path: Path,
) -> None:
    """Save LLM-as-judge evaluation results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for j in judge_results:
        rows.append({
            "recommendation": j.recommendation,
            "predicted_label": j.predicted_label,
            "label_score": j.label_score,
            "justification_score": j.justification_score,
            "evidence_score": j.evidence_score,
            "overall_score": j.overall_score,
            "reasoning": j.reasoning,
        })
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[pipeline] Judge results saved → {output_path}")


# Sub-commands
def cmd_build(args: argparse.Namespace) -> None:
    """Build embedding + BM25 indices from evidence CSVs."""
    if len(args.input) == 1:
        build_index(args.input[0], args.model)
    else:
        build_merged_index(*args.input, model_key=args.model)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate retrieval on gold standard and/or an MTEB benchmark."""
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

    # ── MTEB benchmark ──
    if args.benchmark and args.benchmark.exists():

        print("  MTEB Benchmark Retrieval Evaluation")
        bench_metrics = evaluate_benchmark(
            retriever,
            args.benchmark,
            top_k_retrieve=args.top_k,
            rerank_top=args.rerank_top,
        )
        all_metrics["benchmark"] = bench_metrics
        print(format_retrieval_report(
            bench_metrics, "MTEB Benchmark Retrieval",
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


def cmd_whitepaper(args: argparse.Namespace) -> None:
    """Retrieve + classify whitepaper recommendations (unlabelled)."""
    from evaluation.evaluation import load_whitepaper_recommendations

    retriever = HybridRetriever.from_disk(
        args.model, use_reranker=not args.no_rerank,
    )

    wp_rows = load_whitepaper_recommendations(args.input)
    recs = [
        Recommendation(
            section=r["section"],
            subsection=r["subsection"],
            title=r["title"],
            text=r.get("recommendation", "").strip(),
        )
        for r in wp_rows
    ]
    # Build query from section/title context when recommendation text is empty
    for rec in recs:
        if not rec.text:
            rec.text = f"{rec.section} {rec.subsection} {rec.title}".strip()

    print(f"\n[whitepaper] {len(recs)} recommendations loaded")

    # ── Retrieval-only pass (always runs) ──
    retrieval_results = []
    for i, rec in enumerate(recs, 1):
        result = retriever.retrieve(
            rec.text, top_k=args.top_k, rerank_top=args.rerank_top,
        )
        retrieval_results.append(result)
        if (i) % 10 == 0:
            print(f"  [{i}/{len(recs)}] queries retrieved")
    print(f"[whitepaper] Retrieval complete for {len(recs)} recommendations")

    # ── Optional classification ──
    results: list[ClassificationResult] = []
    if not args.retrieve_only:
        try:
            from rag.classifier import AlignmentClassifier
            classifier = AlignmentClassifier()
        except (EnvironmentError, ImportError) as exc:
            print(
                f"[warn] Classification unavailable ({exc}). "
                "Exporting retrieval-only results."
            )
            classifier = None

        if classifier:
            for i, (rec, ret) in enumerate(
                zip(recs, retrieval_results), 1,
            ):
                print(f"  [{i}/{len(recs)}] classifying: {rec.text[:80]}…")
                classification = classifier.classify(
                    rec.text, ret.ranked_chunks,
                )
                results.append(classification)
                print(f"    → {classification.label}")

    # ── Export CSV for human evaluation ──
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    for idx, rec in enumerate(recs):
        ret = retrieval_results[idx]
        cls = results[idx] if idx < len(results) else None
        rows_out.append({
            "section": rec.section,
            "subsection": rec.subsection,
            "title": rec.title,
            "recommendation_query": rec.text,
            "retrieved_documents": "; ".join(sorted(
                {c.document for c in ret.ranked_chunks}
            )),
            "retrieved_articles": "; ".join(sorted(
                {c.article for c in ret.ranked_chunks if c.article}
            )),
            "top_chunk_ids": "; ".join(c.id for c in ret.ranked_chunks),
            "top_chunk_texts": "\n---\n".join(
                f"[{c.document} | {c.article}]\n{c.text[:400]}" for c in ret.ranked_chunks
            ),
            "alignment_label": cls.label if cls else "",
            "justification": cls.justification if cls else "",
            "cited_chunk_ids": (
                "; ".join(cls.cited_chunk_ids) if cls else ""
            ),
            "human_label": "",
            "human_notes": "",
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[whitepaper] Results saved → {output_path}")

    # ── Save per-query retrieved chunks (full text) ──
    chunks_path = output_path.parent / (output_path.stem + "_retrieved_chunks.csv")
    rec_texts = [rec.text for rec in recs]
    _save_retrieved_chunks_csv(rec_texts, retrieval_results, chunks_path, top_k=args.top_k)

    # ── Optional LLM-as-judge evaluation ──
    if getattr(args, "judge", False) and results:
        try:
            from rag.llm_judge import LLMJudge
            print("\n[whitepaper] Running LLM-as-judge evaluation…")
            judge = LLMJudge()
            judge_results = judge.evaluate_batch(results)
            judge_path = output_path.parent / (output_path.stem + "_judge.csv")
            _save_judge_results_csv(judge_results, judge_path)
        except Exception as exc:
            print(f"[warn] Judge evaluation failed: {exc}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run retrieval evaluation on an MTEB benchmark dataset."""
    retriever = HybridRetriever.from_disk(
        args.model, use_reranker=not args.no_rerank,
    )

    print(f"[benchmark] Evaluating on {args.input}")
    bench_metrics = evaluate_benchmark(
        retriever,
        args.input,
        top_k_retrieve=args.top_k,
        rerank_top=args.rerank_top,
    )
    print(format_retrieval_report(bench_metrics, f"Benchmark: {args.input.name}"))

    if args.output:
        save_metrics_json(bench_metrics, None, args.output)
    print("[benchmark] Done.")


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

    # ── Save per-query retrieved chunks (gold standard queries) ──
    if recs and results:
        gs_queries = [r.recommendation for r in results]
        # We need the retrieval results — re-retrieve them for saving
        # (results already has retrieved_chunks on each ClassificationResult)
        from data_models import RetrievalResult
        gs_ret_results = [
            RetrievalResult(
                query=r.recommendation,
                ranked_chunks=r.retrieved_chunks,
                scores=[0.0] * len(r.retrieved_chunks),
            )
            for r in results
        ]
        chunks_path = (
            args.output.parent / "classified_retrieved_chunks.csv"
            if args.output else OUTPUT_DIR / "classified_retrieved_chunks.csv"
        )
        _save_retrieved_chunks_csv(gs_queries, gs_ret_results, chunks_path, top_k=args.top_k)

    # ── Optional LLM-as-judge ──
    if getattr(args, "judge", False) and results:
        try:
            from rag.llm_judge import LLMJudge
            print("\n[run] Running LLM-as-judge evaluation…")
            judge = LLMJudge()
            judge_results_list = judge.evaluate_batch(results)
            judge_path = (
                args.output.parent / "judge_results.csv"
                if args.output else OUTPUT_DIR / "judge_results.csv"
            )
            _save_judge_results_csv(judge_results_list, judge_path)
        except Exception as exc:
            print(f"[warn] Judge evaluation failed: {exc}")

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
        "evaluate", help="Evaluate retrieval on gold standard / MTEB benchmark",
    )
    p_eval.add_argument(
        "--gold", type=Path, default=GOLD_STANDARD_CSV,
        help="Gold-standard CSV path",
    )
    p_eval.add_argument(
        "--benchmark", type=Path, default=None,
        help="MTEB benchmark JSON path (optional)",
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

    # whitepaper
    p_wp = sub.add_parser(
        "whitepaper",
        help="Retrieve + classify whitepaper recommendations (export for human eval)",
    )
    p_wp.add_argument(
        "-i", "--input", type=Path,
        default=WHITEPAPER_RECOMMENDATIONS_CSV,
        help="Whitepaper recommendations CSV (semicolon-delimited)",
    )
    p_wp.add_argument(
        "-o", "--output", type=Path,
        default=OUTPUT_DIR / "whitepaper_classified.csv",
        help="Output CSV (includes human_label / human_notes columns)",
    )
    p_wp.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )
    p_wp.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_wp.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    p_wp.add_argument("--no-rerank", action="store_true")
    p_wp.add_argument(
        "--retrieve-only", action="store_true",
        help="Skip classification; export retrieval results only",
    )
    p_wp.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-judge on classifications and save scores",
    )

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run retrieval evaluation on an MTEB benchmark (JSON)",
    )
    p_bench.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Benchmark JSON file",
    )
    p_bench.add_argument(
        "-o", "--output", type=Path, default=None,
        help="JSON path for metric export",
    )
    p_bench.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )
    p_bench.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K)
    p_bench.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP)
    p_bench.add_argument("--no-rerank", action="store_true")

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
    p_run.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-judge on classifications and save scores",
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "whitepaper":
        cmd_whitepaper(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
