from __future__ import annotations
import argparse
import gc
from pathlib import Path
from typing import Any

from config import EVIDENCE_CSV, GOLD_STANDARD_CSV, INDEX_DIR, WHITEPAPER_RECOMMENDATIONS_CSV
from data_models import ClassificationResult
from embedding_indexing import build_index, build_merged_index
from evaluation.commands import (
    cmd_download_models,
    cmd_merge_eval,
    cmd_unified_eval,
)
from pipeline_io import (
    load_recommendations,
    save_prompt_output_csv,
    save_retrieved_chunks_csv,
)
from retrieval.retrieval import HybridRetriever

__all__ = [
    "cmd_build",
    "cmd_prompt",
    "cmd_evaluate",
    "cmd_download_models",
    "cmd_merge_eval",
]

def _require_file(path: Path, label: str) -> None:
    """Raise a clear error if a required input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")

def _index_paths(model_key: str) -> tuple[Path, Path, Path]:
    """Return the three index artifact paths for a model key."""
    prefix = INDEX_DIR / model_key
    return (
        Path(str(prefix) + "_faiss.index"),
        Path(str(prefix) + "_bm25.pkl"),
        Path(str(prefix) + "_chunks.pkl"),
    )

def _require_indices(model_key: str) -> None:
    """Validate that all retrieval index artifacts exist for prompt mode."""
    faiss_path, bm25_path, chunks_path = _index_paths(model_key)
    missing = [p for p in (faiss_path, bm25_path, chunks_path) if not p.exists()]
    if not missing:
        return

    missing_str = "\n".join(f"  - {p}" for p in missing)
    raise FileNotFoundError(
        "Missing retrieval indices for prompt mode. Build indices first with:\n"
        f"  python main.py build -i {EVIDENCE_CSV} -m {model_key}\n"
        "Missing files:\n"
        f"{missing_str}"
    )

def _print_progress(stage: str, idx: int, total: int) -> None:
    """Print progress every 10 items and at completion."""
    if idx % 10 == 0 or idx == total:
        print(f"  [{stage}] {idx}/{total}")

def _retrieve_all(
    retriever: Any,
    recs: list[Any],
    top_k: int,
    rerank_top: int,
    retrieval_mode: str,
    inner_retrieval_method: str = "rrf",
) -> list[Any]:
    """Run retrieval for all recommendations with consistent progress output."""
    results: list[Any] = []
    total = len(recs)
    for idx, rec in enumerate(recs, start=1):
        results.append(
            retriever.retrieve(
                rec.text,
                top_k=top_k,
                rerank_top=rerank_top,
                retrieval_mode=retrieval_mode,
                max_chunks_per_doc=2,
                near_dup_suppression=False,
                inner_retrieval_method=inner_retrieval_method,
            )
        )
        _print_progress("retrieve", idx, total)
    return results

def _free_gpu(note: str = "") -> None:
    """Release Python objects and flush the CUDA allocator cache."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        tag = f" [{note}]" if note else ""
        print(f"[gpu]{tag} allocated={alloc:.2f} GiB | reserved={reserved:.2f} GiB")

def _classify_all(recs: list[Any], retrieval_results: list[Any]) -> list[ClassificationResult]:
    """Classify all retrieved recommendation contexts."""
    from rag.classifier import AlignmentClassifier

    classifier = AlignmentClassifier()
    cls_results: list[ClassificationResult] = []
    total = len(recs)
    for idx, (rec, retrieval) in enumerate(zip(recs, retrieval_results), start=1):
        cls_results.append(classifier.classify(rec.text, retrieval.ranked_chunks, title=rec.title))
        _print_progress("classify", idx, total)
    del classifier
    _free_gpu("after classifier")
    return cls_results

def cmd_build(args: argparse.Namespace) -> None:
    """Build index artifacts for one or more evidence CSV inputs."""
    for input_path in args.input:
        _require_file(Path(input_path), "evidence CSV input")

    print(f"[build] Writing indices under: {INDEX_DIR}")
    if len(args.input) == 1:
        build_index(args.input[0], args.model)
    else:
        build_merged_index(*args.input, model_key=args.model)

def cmd_prompt(args: argparse.Namespace) -> None:
    """Retrieve, classify, and optionally run LLM judge."""
    _require_file(Path(args.input), "recommendations CSV")
    _require_indices(args.model)

    print(f"[prompt] Using indices from: {INDEX_DIR}")
    print(f"[prompt] Output file: {args.output}")
    print(f"[prompt] Retrieval mode: {args.retrieval_mode}")

    recs = load_recommendations(args.input)
    print(f"[prompt] Loaded {len(recs)} recommendations")

    retriever = HybridRetriever.from_disk(args.model, use_reranker=True)
    retrieval_results = _retrieve_all(
        retriever,
        recs,
        top_k=args.top_k,
        rerank_top=args.rerank_top,
        retrieval_mode=args.retrieval_mode,
        inner_retrieval_method=getattr(args, "inner_retrieval_method", "rrf"),
    )
    # Free the retriever (embedding model + reranker) before loading the LLM.
    del retriever
    _free_gpu("after retriever")
    cls_results = _classify_all(recs, retrieval_results)
    save_prompt_output_csv(args.output, recs, retrieval_results, cls_results)
    print(f"[prompt] Saved -> {args.output}")

    chunks_path = args.output.parent / f"{args.output.stem}_retrieved_chunks.csv"
    save_retrieved_chunks_csv([r.text for r in recs], retrieval_results, chunks_path, top_k=args.top_k)
    print(f"[prompt] Retrieved chunks -> {chunks_path}")

    if args.judge:
        from rag.llm_judge import LLMJudge

        judge_path = args.output.parent / f"{args.output.stem}_judge.csv"
        if judge_path.exists():
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            judge_path = args.output.parent / f"{args.output.stem}_judge_{ts}.csv"
            print(f"[prompt] Judge output already exists — writing to {judge_path}")

        try:
            judge = LLMJudge()
            judge_results = judge.evaluate_batch(cls_results, output_path=judge_path)
            del judge
            _free_gpu("after judge")
            if not judge_results:
                print("[prompt] Judge produced no results — check logs for errors.")
            else:
                print(f"[prompt] Judge -> {judge_path} ({len(judge_results)} rows)")
        except Exception as exc:
            print(f"[prompt] Judge stage ERROR: {exc}")

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run unified retrieval evaluation."""
    _require_file(GOLD_STANDARD_CSV, "gold-standard CSV")
    _require_file(WHITEPAPER_RECOMMENDATIONS_CSV, "whitepaper recommendations CSV")
    _require_file(Path(args.evidence_csv), "evidence CSV for auto-build")

    print(f"[evaluate] Unified outputs directory: {args.output_dir}")
    cmd_unified_eval(args)