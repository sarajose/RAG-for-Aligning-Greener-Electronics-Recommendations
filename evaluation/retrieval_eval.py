"""Unified retrieval evaluation orchestrator.
All results are written to metrics_all.csv in the output directory.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    GOLD_STANDARD_CSV,
    RRF_K,
    SPLADE_MAX_LENGTH,
    SPLADE_MODEL,
)
from data_models import RetrievalResult
from embedding_indexing import build_index, get_embed_model, load_indices
from evaluation.evaluation import (
    _indices_exist,
    _metrics_to_rows,
    evaluate_retrieval,
)
from evaluation.mteb_eval import (
    _build_mteb_retriever,
    _build_mteb_splade_retriever,
    _evaluate_mteb_chunk_level,
)
from retrieval.base_retriever import BaseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever as CompositeHybridRetriever
from retrieval.reranker import Reranker, RerankedRetriever
from retrieval.retrieval import HybridRetriever as FullHybridRetriever
from retrieval.splade_retriever import SPLADERetriever

MTEB_DATASET = "mteb/MuPLeR-retrieval"
MTEB_SPLIT = "test"
MTEB_EMBED_BATCH_SIZE = 32
MTEB_DEVICE = "auto"
MTEB_PRECISION = "float32"

class _SplitEvidenceRetriever(BaseRetriever):
    """Wraps FullHybridRetriever with retrieval_mode fixed to split_evidence_retrieval."""

    def __init__(self, full_retriever: FullHybridRetriever) -> None:
        self._retriever = full_retriever

    @property
    def name(self) -> str:
        return "Hybrid (BM25 + FAISS + RRF, split_evidence)"

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, **_kwargs) -> RetrievalResult:
        """Delegate to FullHybridRetriever with split_evidence_retrieval mode."""
        return self._retriever.retrieve(query, top_k=top_k, retrieval_mode="split_evidence_retrieval")

def _build_retrievers_for_model(
    model_key: str,
    reranker: Reranker | None,
    top_k: int,
    rerank_top: int,
    rrf_k: int = RRF_K,
    retrieval_mode: str = "flat_baseline",
) -> dict[str, Any]:
    """Build BM25, dense, hybrid, and reranked retriever variants for one embedding model."""
    faiss_index, bm25_index, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)

    bm25 = BM25Retriever(bm25_index, chunks)
    dense = DenseRetriever(faiss_index, chunks, embed_model)
    if retrieval_mode == "split_evidence_retrieval":
        full_retriever = FullHybridRetriever(faiss_index, bm25_index, chunks, embed_model, use_reranker=False)
        hybrid = _SplitEvidenceRetriever(full_retriever)
    else:
        hybrid = CompositeHybridRetriever(faiss_index, bm25_index, chunks, embed_model, rrf_k=rrf_k)

    retrievers: dict[str, Any] = {"bm25": bm25, "dense": dense, "rrf": hybrid}
    if reranker is not None:
        initial_k = max(top_k * 2, 30)
        retrievers["bm25_rerank"]  = RerankedRetriever(bm25,   reranker, initial_k=initial_k, final_k=rerank_top)
        retrievers["dense_rerank"] = RerankedRetriever(dense,  reranker, initial_k=initial_k, final_k=rerank_top)
        retrievers["rrf_rerank"]   = RerankedRetriever(hybrid, reranker, initial_k=initial_k, final_k=rerank_top)
    return retrievers

def _faiss_preflight_ok(model_key: str) -> tuple[bool, str]:
    """Sanity-check the FAISS index for a model: dimension, size, and a probe query."""
    try:
        faiss_index, _, chunks = load_indices(model_key)
        ntotal = int(getattr(faiss_index, "ntotal", -1))
        dim    = int(getattr(faiss_index, "d", 0))
        if dim <= 0:
            return False, "FAISS index dimension is invalid"
        if ntotal <= 0:
            return False, "FAISS index is empty"
        if ntotal != len(chunks):
            return False, f"FAISS/chunks mismatch (ntotal={ntotal}, chunks={len(chunks)})"
        _, idx = faiss_index.search(np.zeros((1, dim), dtype=np.float32), 1)
        if idx.shape != (1, 1):
            return False, "FAISS probe query returned unexpected shape"
        return True, f"ntotal={ntotal}, dim={dim}"
    except Exception as exc:
        return False, str(exc)

def _run_splade_eval(
    args: argparse.Namespace,
    reranker,
    metrics_rows: list[dict],
) -> None:
    """Evaluate SPLADE sparse baseline and append metric rows in-place."""
    print("\n=== SPLADE baseline ===")
    base_model_for_chunks = args.models[0] if args.models else next(iter(EMBEDDING_MODELS))
    if not _indices_exist(base_model_for_chunks):
        print(f"[build] Missing indices for {base_model_for_chunks}; building now for SPLADE chunks...")
        build_index(args.evidence_csv, base_model_for_chunks)

    splade_base = SPLADERetriever.from_disk(
        model_key=base_model_for_chunks, model_name=SPLADE_MODEL, max_length=SPLADE_MAX_LENGTH,
    )
    splade_retrievers: dict[str, Any] = {"splade": splade_base}
    if reranker is not None:
        splade_retrievers["splade_rerank"] = RerankedRetriever(
            splade_base, reranker, initial_k=max(args.top_k * 2, 30), final_k=args.rerank_top,
        )

    for method_name, retriever in splade_retrievers.items():
        print(f"[gold-doc] Evaluating {method_name} ...")
        metrics_by_k = evaluate_retrieval(
            retriever, gold_path=GOLD_STANDARD_CSV,
            k_values=sorted(set(args.k_values)), top_k_retrieve=max(args.top_k * 3, 30), rerank_top=args.rerank_top,
        )
        metrics_rows.extend(_metrics_to_rows(
            metrics_by_k, dataset="gold_standard", level="document", model_key="splade", method=method_name,
        ))
        if int(getattr(next(iter(metrics_by_k.values())), "num_queries", 0)) == 0:
            raise RuntimeError("Gold-standard evaluation produced 0 queries.")

    try:
        base_mteb_splade = _build_mteb_splade_retriever(
            dataset_id=MTEB_DATASET, max_corpus=None, model_name=SPLADE_MODEL, max_length=SPLADE_MAX_LENGTH,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
            print(f"[warn] Could not build MTEB SPLADE retriever due to memory limits: {exc}")
            return
        raise

    mteb_eval_map: dict[str, Any] = {"splade": base_mteb_splade}
    if reranker is not None:
        mteb_eval_map["splade_rerank"] = RerankedRetriever(
            base_mteb_splade, reranker, initial_k=max(args.top_k * 2, 30), final_k=args.rerank_top,
        )

    for mteb_method in splade_retrievers:
        mteb_out_csv = args.output_dir / f"mteb_retrieved_chunks_splade_{mteb_method}.csv"
        try:
            print(f"[mteb] Starting SPLADE chunk-level eval: method={mteb_method}", flush=True)
            mteb_metrics = _evaluate_mteb_chunk_level(
                retriever=mteb_eval_map[mteb_method], dataset_id=MTEB_DATASET, split_name=MTEB_SPLIT,
                k_values=sorted(set(args.k_values)), top_k=max(args.top_k * 3, 30), max_corpus=None,
                model_key="splade", method=mteb_method, out_retrieved_csv=mteb_out_csv,
            )
            metrics_rows.extend(_metrics_to_rows(
                mteb_metrics, dataset="mteb_legal", level="chunk", model_key="splade", method=mteb_method,
            ))
            print(f"[mteb] Finished SPLADE chunk-level eval: method={mteb_method}", flush=True)
        except Exception as exc:
            msg = str(exc).lower()
            if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
                print(f"[warn] Skipping MTEB for model=splade method={mteb_method} due to memory limits: {exc}")
            else:
                raise

def cmd_unified_eval(args: argparse.Namespace) -> None:
    """Run unified evaluation: gold standard, MTEB, and SPLADE; write metrics_all.csv."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[setup] CUDA available: True (count={torch.cuda.device_count()}) | using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[setup] CUDA available: False | running on CPU.")
    except Exception as exc:
        print(f"[setup] Could not probe CUDA status: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    metrics_csv = args.output_dir / "metrics_all.csv"

    print("[setup] Loading cross-encoder reranker...")
    reranker = None
    try:
        reranker = Reranker()
    except Exception as exc:
        msg = str(exc).lower()
        if "outofmemory" in msg or "cuda out of memory" in msg:
            print("[warn] CUDA OOM while loading reranker; continuing without reranker.")
        else:
            raise

    for model_key in args.models:
        print(f"\n=== Model: {model_key} ===")

        if not _indices_exist(model_key):
            print(f"[build] Missing indices for {model_key}; building now...")
            build_index(args.evidence_csv, model_key)

        ok, reason = _faiss_preflight_ok(model_key)
        if ok:
            print(f"[sanity] FAISS preflight OK for {model_key}: {reason}")
        else:
            print(f"[warn] FAISS preflight failed for {model_key}: {reason}. Rebuilding...")
            build_index(args.evidence_csv, model_key)
            ok_after, reason_after = _faiss_preflight_ok(model_key)
            if not ok_after:
                raise RuntimeError(f"FAISS preflight still failing after rebuild for '{model_key}': {reason_after}")
            print(f"[sanity] FAISS preflight recovered for {model_key}: {reason_after}")

        retrievers = _build_retrievers_for_model(
            model_key, reranker=reranker, top_k=args.top_k, rerank_top=args.rerank_top,
            rrf_k=RRF_K, retrieval_mode=getattr(args, "retrieval_mode", "flat_baseline"),
        )

        # Gold standard evaluation
        for method_name, retriever in retrievers.items():
            print(f"[gold-doc] Evaluating {method_name} ...")
            metrics_by_k = evaluate_retrieval(
                retriever, gold_path=GOLD_STANDARD_CSV, k_values=sorted(set(args.k_values)),
                top_k_retrieve=max(args.top_k * 3, 30), rerank_top=args.rerank_top,
            )
            metrics_rows.extend(_metrics_to_rows(
                metrics_by_k, dataset="gold_standard", level="document", model_key=model_key, method=method_name,
            ))
            if int(getattr(next(iter(metrics_by_k.values())), "num_queries", 0)) == 0:
                raise RuntimeError("Gold-standard evaluation produced 0 queries. Check gold CSV path/format.")

        # MTEB chunk-level evaluation
        for mteb_method in list(retrievers.keys()):
            mteb_out_csv = args.output_dir / f"mteb_retrieved_chunks_{model_key}_{mteb_method}.csv"
            try:
                print(f"[mteb] Starting chunk-level eval: model={model_key}, method={mteb_method}", flush=True)
                mteb_retriever = _build_mteb_retriever(
                    model_key=model_key, method=mteb_method, reranker=reranker,
                    dataset_id=MTEB_DATASET, max_corpus=None, embed_batch_size=MTEB_EMBED_BATCH_SIZE,
                    embed_device=MTEB_DEVICE, embed_precision=MTEB_PRECISION,
                )
                mteb_metrics = _evaluate_mteb_chunk_level(
                    retriever=mteb_retriever, dataset_id=MTEB_DATASET, split_name=MTEB_SPLIT,
                    k_values=sorted(set(args.k_values)), top_k=max(args.top_k * 3, 30), max_corpus=None,
                    model_key=model_key, method=mteb_method, out_retrieved_csv=mteb_out_csv,
                )
                metrics_rows.extend(_metrics_to_rows(
                    mteb_metrics, dataset="mteb_legal", level="chunk", model_key=model_key, method=mteb_method,
                ))
                print(f"[mteb] Finished chunk-level eval: model={model_key}, method={mteb_method}", flush=True)
            except Exception as exc:
                msg = str(exc).lower()
                if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
                    print(f"[warn] Skipping MTEB for model={model_key} method={mteb_method} due to memory limits: {exc}")
                else:
                    raise

    # SPLADE baseline
    _run_splade_eval(args, reranker, metrics_rows)

    # Write final metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n[done] Unified evaluation finished.")
    print(f"[done] Metrics: {metrics_csv}")
