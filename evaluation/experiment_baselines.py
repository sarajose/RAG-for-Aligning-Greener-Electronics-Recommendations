"""SPLADE and ColBERT baseline evaluation helpers."""

from __future__ import annotations

import argparse
from typing import Any

from config import EMBEDDING_MODELS
from embedding_indexing import build_index
from evaluation.evaluation import evaluate_retrieval
from evaluation.experiment_exports import export_gold_retrieved_chunks, export_whitepaper_retrieved_chunks
from evaluation.experiment_helpers import _indices_exist, _metrics_to_rows
from retrieval.colbert_retriever import ColBERTRetriever
from retrieval.reranker import RerankedRetriever
from retrieval.splade_retriever import SPLADERetriever


def _run_splade_eval(
    args: argparse.Namespace,
    reranker,
    metrics_rows: list[dict],
    checkpoint_fn,
) -> None:
    """Evaluate SPLADE baseline and append metric rows in-place."""
    from evaluation.experiment_mteb import _build_mteb_splade_retriever, _evaluate_mteb_chunk_level

    print("\n=== SPLADE baseline ===")
    base_model_for_chunks = args.models[0] if args.models else next(iter(EMBEDDING_MODELS))
    if not _indices_exist(base_model_for_chunks):
        if not args.auto_build_indices:
            raise FileNotFoundError(
                "SPLADE needs chunk artifacts from at least one built index. "
                f"Missing indices for '{base_model_for_chunks}'. "
                "Build once with --auto-build-indices or main.py build."
            )
        print(f"[build] Missing indices for {base_model_for_chunks}; building now for SPLADE chunks...")
        build_index(args.evidence_csv, base_model_for_chunks)

    splade_base = SPLADERetriever.from_disk(
        model_key=base_model_for_chunks,
        model_name=args.splade_model,
        max_length=args.splade_max_length,
    )
    splade_retrievers: dict[str, Any] = {"splade": splade_base}
    if reranker is not None:
        splade_retrievers["splade_rerank"] = RerankedRetriever(
            splade_base,
            reranker,
            initial_k=max(args.top_k * 2, 30),
            final_k=args.rerank_top,
        )

    for method_name, retriever in splade_retrievers.items():
        print(f"[gold-doc] Evaluating {method_name} ...")
        metrics_by_k = evaluate_retrieval(
            retriever,
            gold_path=args.gold_csv,
            k_values=sorted(set(args.k_values)),
            top_k_retrieve=max(args.top_k * 3, 30),
            rerank_top=args.rerank_top,
        )
        metrics_rows.extend(
            _metrics_to_rows(
                metrics_by_k,
                dataset="gold_standard",
                level="document",
                model_key="splade",
                method=method_name,
            )
        )
        any_metric = next(iter(metrics_by_k.values()))
        if int(getattr(any_metric, "num_queries", 0)) == 0:
            raise RuntimeError(
                "Gold-standard evaluation produced 0 queries. "
                "Check gold CSV delimiter/columns and input path."
            )
        checkpoint_fn(f"gold eval splade/{method_name}")

    export_method = "splade_rerank" if "splade_rerank" in splade_retrievers else "splade"
    export_gold_retrieved_chunks(
        retriever=splade_retrievers[export_method],
        model_key="splade",
        method=export_method,
        gold_csv=args.gold_csv,
        out_csv=args.output_dir / f"gold_retrieved_chunks_splade_{export_method}.csv",
        top_k=args.export_k,
    )

    if not args.skip_whitepaper:
        export_whitepaper_retrieved_chunks(
            retriever=splade_retrievers[export_method],
            model_key="splade",
            method=export_method,
            whitepaper_csv=args.whitepaper_csv,
            out_csv=args.output_dir / f"whitepaper_retrieved_chunks_splade_{export_method}.csv",
            top_k=args.export_k,
        )

    if not args.skip_mteb:
        mteb_method = "splade_rerank" if reranker is not None else "splade"
        try:
            print(
                f"[mteb] Starting chunk-level eval for model=splade, method={mteb_method}, "
                f"dataset={args.mteb_dataset}, split={args.mteb_split}",
                flush=True,
            )
            mteb_retriever = _build_mteb_splade_retriever(
                dataset_id=args.mteb_dataset,
                max_corpus=args.max_corpus,
                model_name=args.splade_model,
                max_length=args.splade_max_length,
            )
            if reranker is not None:
                mteb_retriever = RerankedRetriever(
                    mteb_retriever,
                    reranker,
                    initial_k=max(args.top_k * 2, 30),
                    final_k=args.rerank_top,
                )
            mteb_metrics = _evaluate_mteb_chunk_level(
                retriever=mteb_retriever,
                dataset_id=args.mteb_dataset,
                split_name=args.mteb_split,
                k_values=sorted(set(args.k_values)),
                top_k=max(args.top_k * 3, 30),
                max_corpus=args.max_corpus,
                model_key="splade",
                method=mteb_method,
                out_retrieved_csv=args.output_dir / f"mteb_retrieved_chunks_splade_{mteb_method}.csv",
            )
            metrics_rows.extend(
                _metrics_to_rows(
                    mteb_metrics,
                    dataset="mteb_legalbench",
                    level="chunk",
                    model_key="splade",
                    method=mteb_method,
                )
            )
            checkpoint_fn(f"mteb eval splade/{mteb_method}")
            print(
                f"[mteb] Finished chunk-level eval for model=splade, method={mteb_method}",
                flush=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
                print(f"[warn] Skipping MTEB for model=splade due to memory limits: {exc}")
            else:
                raise


def _run_colbert_eval(
    args: argparse.Namespace,
    reranker,
    metrics_rows: list[dict],
    checkpoint_fn,
) -> None:
    """Evaluate ColBERT baseline and append metric rows in-place."""
    print("\n=== BGE-M3 ColBERT multi-vector baseline ===")
    base_model_for_chunks = args.models[0] if args.models else next(iter(EMBEDDING_MODELS))
    if not _indices_exist(base_model_for_chunks):
        raise FileNotFoundError(
            f"ColBERT needs chunk artifacts from '{base_model_for_chunks}'. "
            "Build with main.py build or --auto-build-indices."
        )
    colbert_base = ColBERTRetriever.from_disk(model_key=base_model_for_chunks)
    colbert_retrievers: dict[str, Any] = {"colbert": colbert_base}
    if reranker is not None:
        colbert_retrievers["colbert_rerank"] = RerankedRetriever(
            colbert_base, reranker,
            initial_k=max(args.top_k * 2, 30),
            final_k=args.rerank_top,
        )

    for method_name, retriever in colbert_retrievers.items():
        print(f"[gold-doc] Evaluating {method_name} ...")
        metrics_by_k = evaluate_retrieval(
            retriever,
            gold_path=args.gold_csv,
            k_values=sorted(set(args.k_values)),
            top_k_retrieve=max(args.top_k * 3, 30),
            rerank_top=args.rerank_top,
        )
        metrics_rows.extend(
            _metrics_to_rows(
                metrics_by_k,
                dataset="gold_standard",
                level="document",
                model_key="colbert",
                method=method_name,
            )
        )
        any_metric = next(iter(metrics_by_k.values()))
        if int(getattr(any_metric, "num_queries", 0)) == 0:
            raise RuntimeError(
                "Gold-standard evaluation produced 0 queries. "
                "Check gold CSV delimiter/columns and input path."
            )
        checkpoint_fn(f"gold eval colbert/{method_name}")

    export_method = "colbert_rerank" if "colbert_rerank" in colbert_retrievers else "colbert"
    export_gold_retrieved_chunks(
        retriever=colbert_retrievers[export_method],
        model_key="colbert",
        method=export_method,
        gold_csv=args.gold_csv,
        out_csv=args.output_dir / f"gold_retrieved_chunks_colbert_{export_method}.csv",
        top_k=args.export_k,
    )

    if not args.skip_whitepaper:
        export_whitepaper_retrieved_chunks(
            retriever=colbert_retrievers[export_method],
            model_key="colbert",
            method=export_method,
            whitepaper_csv=args.whitepaper_csv,
            out_csv=args.output_dir / f"whitepaper_retrieved_chunks_colbert_{export_method}.csv",
            top_k=args.export_k,
        )
