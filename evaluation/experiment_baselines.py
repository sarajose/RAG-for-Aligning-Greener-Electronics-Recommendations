"""SPLADE baseline evaluation helper."""

from __future__ import annotations

import argparse
from typing import Any

from config import EMBEDDING_MODELS, GOLD_STANDARD_CSV, SPLADE_MAX_LENGTH, SPLADE_MODEL, WHITEPAPER_RECOMMENDATIONS_CSV
from embedding_indexing import build_index
from evaluation.evaluation import evaluate_retrieval
from evaluation.experiment_exports import export_gold_retrieved_chunks, export_whitepaper_retrieved_chunks
from evaluation.experiment_helpers import _indices_exist, _metrics_to_rows
from retrieval.reranker import RerankedRetriever
from retrieval.splade_retriever import SPLADERetriever

MTEB_DATASET = "mteb/MuPLeR-retrieval"
MTEB_SPLIT = "test"
EXPORT_K = 10


def _run_splade_eval(
    args: argparse.Namespace,
    reranker,
    metrics_rows: list[dict],
    checkpoint_fn,
    has_metrics_fn=None,
    step_done_fn=None,
    mark_step_fn=None,
) -> None:
    """Evaluate SPLADE baseline and append metric rows in-place."""
    from evaluation.experiment_mteb import _build_mteb_splade_retriever, _evaluate_mteb_chunk_level

    print("\n=== SPLADE baseline ===")
    base_model_for_chunks = args.models[0] if args.models else next(iter(EMBEDDING_MODELS))
    if not _indices_exist(base_model_for_chunks):
        print(f"[build] Missing indices for {base_model_for_chunks}; building now for SPLADE chunks...")
        build_index(args.evidence_csv, base_model_for_chunks)

    splade_base = SPLADERetriever.from_disk(
        model_key=base_model_for_chunks,
        model_name=SPLADE_MODEL,
        max_length=SPLADE_MAX_LENGTH,
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
        step_key = f"gold_doc__splade__{method_name}"
        if has_metrics_fn is not None and has_metrics_fn("gold_standard", "document", "splade", method_name):
            print(f"[resume] Skipping SPLADE gold-doc step already in metrics: {method_name}")
            if mark_step_fn is not None:
                mark_step_fn(step_key)
            continue
        print(f"[gold-doc] Evaluating {method_name} ...")
        metrics_by_k = evaluate_retrieval(
            retriever,
            gold_path=GOLD_STANDARD_CSV,
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
        if mark_step_fn is not None:
            mark_step_fn(step_key)

    export_method = "splade_rerank" if "splade_rerank" in splade_retrievers else "splade"
    gold_out_csv = args.output_dir / f"gold_retrieved_chunks_splade_{export_method}.csv"
    gold_export_step = f"export_gold__splade__{export_method}"
    if step_done_fn is not None and step_done_fn(gold_export_step, [gold_out_csv]):
        print(f"[resume] Skipping SPLADE gold export (already done): {gold_out_csv}")
    else:
        export_gold_retrieved_chunks(
            retriever=splade_retrievers[export_method],
            model_key="splade",
            method=export_method,
            gold_csv=GOLD_STANDARD_CSV,
            out_csv=gold_out_csv,
            top_k=EXPORT_K,
        )
        if mark_step_fn is not None:
            mark_step_fn(gold_export_step)

    whitepaper_out_csv = args.output_dir / f"whitepaper_retrieved_chunks_splade_{export_method}.csv"
    whitepaper_export_step = f"export_whitepaper__splade__{export_method}"
    if step_done_fn is not None and step_done_fn(whitepaper_export_step, [whitepaper_out_csv]):
        print(f"[resume] Skipping SPLADE whitepaper export (already done): {whitepaper_out_csv}")
    else:
        export_whitepaper_retrieved_chunks(
            retriever=splade_retrievers[export_method],
            model_key="splade",
            method=export_method,
            whitepaper_csv=WHITEPAPER_RECOMMENDATIONS_CSV,
            out_csv=whitepaper_out_csv,
            top_k=EXPORT_K,
        )
        if mark_step_fn is not None:
            mark_step_fn(whitepaper_export_step)

    # Determine which SPLADE MTEB methods still need evaluation
    candidate_mteb_methods = list(splade_retrievers.keys())
    pending_mteb = []
    for _m in candidate_mteb_methods:
        _step = f"mteb_chunk__splade__{_m}"
        _csv = args.output_dir / f"mteb_retrieved_chunks_splade_{_m}.csv"
        if (
            has_metrics_fn is not None
            and has_metrics_fn("mteb_legal", "chunk", "splade", _m)
            and step_done_fn is not None
            and step_done_fn(_step, [_csv])
        ):
            print(f"[resume] Skipping SPLADE MTEB step already completed: {_m}")
        else:
            pending_mteb.append(_m)

    if not pending_mteb:
        return

    # Build SPLADE retriever once for MTEB corpus (no caching available)
    try:
        base_mteb_splade = _build_mteb_splade_retriever(
            dataset_id=MTEB_DATASET,
            max_corpus=None,
            model_name=SPLADE_MODEL,
            max_length=SPLADE_MAX_LENGTH,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
            print(f"[warn] Could not build MTEB SPLADE retriever due to memory limits: {exc}")
            return
        else:
            raise

    mteb_eval_map: dict[str, Any] = {"splade": base_mteb_splade}
    if reranker is not None:
        mteb_eval_map["splade_rerank"] = RerankedRetriever(
            base_mteb_splade,
            reranker,
            initial_k=max(args.top_k * 2, 30),
            final_k=args.rerank_top,
        )

    for mteb_method in pending_mteb:
        mteb_step = f"mteb_chunk__splade__{mteb_method}"
        mteb_out_csv = args.output_dir / f"mteb_retrieved_chunks_splade_{mteb_method}.csv"
        try:
            print(
                f"[mteb] Starting chunk-level eval for model=splade, method={mteb_method}, "
                f"dataset={MTEB_DATASET}, split={MTEB_SPLIT}",
                flush=True,
            )
            mteb_metrics = _evaluate_mteb_chunk_level(
                retriever=mteb_eval_map[mteb_method],
                dataset_id=MTEB_DATASET,
                split_name=MTEB_SPLIT,
                k_values=sorted(set(args.k_values)),
                top_k=max(args.top_k * 3, 30),
                max_corpus=None,
                model_key="splade",
                method=mteb_method,
                out_retrieved_csv=mteb_out_csv,
            )
            metrics_rows.extend(
                _metrics_to_rows(
                    mteb_metrics,
                    dataset="mteb_legal",
                    level="chunk",
                    model_key="splade",
                    method=mteb_method,
                )
            )
            checkpoint_fn(f"mteb eval splade/{mteb_method}")
            if mark_step_fn is not None:
                mark_step_fn(mteb_step)
            print(
                f"[mteb] Finished chunk-level eval for model=splade, method={mteb_method}",
                flush=True,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
                print(f"[warn] Skipping MTEB for model=splade method={mteb_method} due to memory limits: {exc}")
            else:
                raise
