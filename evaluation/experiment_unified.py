"""Unified evaluation command and export helpers."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_RERANK_TOP, DEFAULT_TOP_K, EMBEDDING_MODELS, SPLADE_MAX_LENGTH, SPLADE_MODEL
from embedding_indexing import build_index, load_indices
from evaluation.evaluation import evaluate_retrieval
from evaluation.experiment_exports import export_gold_retrieved_chunks, export_whitepaper_retrieved_chunks
from evaluation.full_eval import add_significance_markers, build_ablation_table, collect_per_query_scores, format_ablation_report
from evaluation.experiment_helpers import (
    _build_metrics_summary_tables,
    _build_retrievers_for_model,
    _indices_exist,
    _metrics_to_rows,
    _validate_ranking_consistency,
)
from evaluation.experiment_mteb import (
    _build_mteb_retriever,
    _build_mteb_splade_retriever,
    _evaluate_mteb_chunk_level,
)
from retrieval.reranker import RerankedRetriever, Reranker
from evaluation.experiment_baselines import _run_splade_eval, _run_colbert_eval


def cmd_unified_eval(args: argparse.Namespace) -> None:
    """Run unified evaluation across gold, MTEB, and whitepaper exports."""
    if args.full_mteb:
        args.max_corpus = None
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        if args.force_cpu:
            print("[setup] force_cpu=True; CUDA disabled for this run.")
        else:
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                print(f"[setup] CUDA available: True (count={cuda_count}) | using GPU: {device_name}")
            else:
                print("[setup] CUDA available: False | running on CPU.")
    except Exception as exc:
        print(f"[setup] Could not probe CUDA status: {exc}")

    mteb_embed_batch_size = max(1, int(getattr(args, "mteb_embed_batch_size", 32)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    metrics_csv = args.output_dir / "metrics_all.csv"

    metric_key_cols = ["dataset", "level", "model_key", "method", "k"]

    def _checkpoint_path(step_key: str) -> Any:
        safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in step_key)
        return checkpoints_dir / f"{safe}.done"

    def _write_step_checkpoint(step_key: str, extra_payload: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "step": step_key,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra_payload:
            payload.update(extra_payload)
        _checkpoint_path(step_key).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _step_done(step_key: str, required_files: list[Any] | None = None) -> bool:
        done = _checkpoint_path(step_key).exists()
        if not done:
            return False
        if not required_files:
            return True
        return all(Path(p).exists() for p in required_files)

    def _has_metrics(dataset: str, level: str, model_key: str, method: str) -> bool:
        rows = [
            r for r in metrics_rows
            if str(r.get("dataset")) == dataset
            and str(r.get("level")) == level
            and str(r.get("model_key")) == model_key
            and str(r.get("method")) == method
        ]
        if not rows:
            return False
        found_k = {int(r.get("k")) for r in rows if r.get("k") is not None}
        expected_k = {int(k) for k in set(args.k_values)}
        return expected_k.issubset(found_k)

    if metrics_csv.exists():
        prev_df = pd.read_csv(metrics_csv)
        if not prev_df.empty:
            metrics_rows = prev_df.to_dict(orient="records")
            print(f"[resume] Loaded {len(metrics_rows)} existing metric rows from {metrics_csv}")

    def _checkpoint_metrics(stage: str) -> None:
        if not metrics_rows:
            return
        dedup_df = pd.DataFrame(metrics_rows)
        dedup_df = dedup_df.drop_duplicates(subset=metric_key_cols, keep="last").reset_index(drop=True)
        metrics_rows.clear()
        metrics_rows.extend(dedup_df.to_dict(orient="records"))
        dedup_df.to_csv(metrics_csv, index=False)
        print(
            f"[checkpoint] Saved {len(metrics_rows)} metric rows after {stage} -> {metrics_csv}",
            flush=True,
        )

    summary: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "k_values": sorted(set(args.k_values)),
        "mteb_dataset": args.mteb_dataset,
        "mteb_split": args.mteb_split,
        "mteb_max_corpus": args.max_corpus,
        "mteb_embed_batch_size": mteb_embed_batch_size,
        "full_mteb": bool(args.full_mteb),
        "top_k": args.top_k,
        "rerank_top": args.rerank_top,
        "export_k": args.export_k,
        "skip_reranker": bool(args.skip_reranker),
        "skip_mteb": bool(args.skip_mteb),
        "skip_whitepaper": bool(args.skip_whitepaper),
        "force_cpu": bool(args.force_cpu),
        "auto_build_indices": bool(args.auto_build_indices),
        "include_splade": bool(getattr(args, "include_splade", False)),
        "splade_model": str(getattr(args, "splade_model", SPLADE_MODEL)),
        "splade_max_length": int(getattr(args, "splade_max_length", SPLADE_MAX_LENGTH)),
        "checkpoints_dir": str(checkpoints_dir),
        "outputs": {},
    }

    reranker = None
    if not args.skip_reranker:
        print("[setup] Loading cross-encoder reranker...")
        try:
            reranker = Reranker()
        except Exception as exc:
            msg = str(exc).lower()
            if "outofmemory" in msg or "cuda out of memory" in msg:
                print("[warn] CUDA OOM while loading reranker; continuing with --skip-reranker behavior.")
                reranker = None
            else:
                raise

    for model_key in args.models:
        print(f"\n=== Model: {model_key} ===")
        if not _indices_exist(model_key):
            if not args.auto_build_indices:
                raise FileNotFoundError(
                    f"Missing indices for model '{model_key}'. Build them first with "
                    f"'python main.py build -i {args.evidence_csv} -m {model_key}' "
                    "or rerun with --auto-build-indices."
                )
            print(f"[build] Missing indices for {model_key}; building now...")
            build_index(args.evidence_csv, model_key)
            _write_step_checkpoint(f"indices_ready__{model_key}")
        else:
            _write_step_checkpoint(f"indices_ready__{model_key}")

        def _faiss_preflight_ok(current_model_key: str) -> tuple[bool, str]:
            try:
                faiss_index, _, chunks = load_indices(current_model_key)
                ntotal = int(getattr(faiss_index, "ntotal", -1))
                dim = int(getattr(faiss_index, "d", 0))
                if dim <= 0:
                    return False, "FAISS index dimension is invalid"
                if ntotal <= 0:
                    return False, "FAISS index is empty"
                if ntotal != len(chunks):
                    return False, f"FAISS/chunks mismatch (ntotal={ntotal}, chunks={len(chunks)})"
                probe_k = 1
                probe_query = np.zeros((1, dim), dtype=np.float32)
                _, idx = faiss_index.search(probe_query, probe_k)
                if idx.shape != (1, probe_k):
                    return False, "FAISS probe query returned unexpected shape"
                return True, f"ntotal={ntotal}, dim={dim}"
            except Exception as exc:
                return False, str(exc)

        ok, reason = _faiss_preflight_ok(model_key)
        if ok:
            print(f"[sanity] FAISS preflight OK for {model_key}: {reason}")
            _write_step_checkpoint(f"faiss_preflight__{model_key}", {"status": "ok", "details": reason})
        else:
            print(f"[warn] FAISS preflight failed for {model_key}: {reason}")
            if not args.auto_build_indices:
                raise RuntimeError(
                    f"FAISS preflight failed for '{model_key}'. Details: {reason}. "
                    "Rebuild with `python main.py build -i <evidence_csv> -m <model_key>` "
                    "or rerun evaluation with --auto-build-indices."
                )
            print(f"[sanity] Rebuilding indices for {model_key} due to failed preflight...")
            build_index(args.evidence_csv, model_key)
            ok_after, reason_after = _faiss_preflight_ok(model_key)
            if not ok_after:
                raise RuntimeError(
                    f"FAISS preflight still failing after rebuild for '{model_key}': {reason_after}"
                )
            print(f"[sanity] FAISS preflight recovered for {model_key}: {reason_after}")
            _write_step_checkpoint(
                f"faiss_preflight__{model_key}",
                {"status": "recovered_after_rebuild", "details": reason_after},
            )

        retrievers = _build_retrievers_for_model(
            model_key,
            reranker=reranker,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
            rrf_k=getattr(args, "rrf_k", 60),
        )

        for method_name, retriever in retrievers.items():
            gold_step = f"gold_doc__{model_key}__{method_name}"
            if _has_metrics("gold_standard", "document", model_key, method_name):
                print(f"[resume] Skipping already-computed gold-doc metrics: {model_key}/{method_name}")
                _write_step_checkpoint(gold_step)
                continue
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
                    model_key=model_key,
                    method=method_name,
                )
            )
            any_metric = next(iter(metrics_by_k.values()))
            if int(getattr(any_metric, "num_queries", 0)) == 0:
                raise RuntimeError(
                    "Gold-standard evaluation produced 0 queries. "
                    "Check gold CSV delimiter/columns and input path."
                )
            _checkpoint_metrics(f"gold eval {model_key}/{method_name}")
            _write_step_checkpoint(gold_step)

        export_method = "rrf_rerank" if "rrf_rerank" in retrievers else "rrf"
        gold_export_csv = args.output_dir / f"gold_retrieved_chunks_{model_key}_{export_method}.csv"
        gold_export_step = f"export_gold__{model_key}__{export_method}"
        if _step_done(gold_export_step, [gold_export_csv]):
            print(f"[resume] Skipping existing gold export: {gold_export_csv}")
        else:
            export_gold_retrieved_chunks(
                retriever=retrievers[export_method],
                model_key=model_key,
                method=export_method,
                gold_csv=args.gold_csv,
                out_csv=gold_export_csv,
                top_k=args.export_k,
            )
            _write_step_checkpoint(gold_export_step, {"artifact": str(gold_export_csv)})

        if not args.skip_whitepaper:
            whitepaper_export_csv = args.output_dir / f"whitepaper_retrieved_chunks_{model_key}_{export_method}.csv"
            whitepaper_export_step = f"export_whitepaper__{model_key}__{export_method}"
            if _step_done(whitepaper_export_step, [whitepaper_export_csv]):
                print(f"[resume] Skipping existing whitepaper export: {whitepaper_export_csv}")
            else:
                export_whitepaper_retrieved_chunks(
                    retriever=retrievers[export_method],
                    model_key=model_key,
                    method=export_method,
                    whitepaper_csv=args.whitepaper_csv,
                    out_csv=whitepaper_export_csv,
                    top_k=args.export_k,
                )
                _write_step_checkpoint(whitepaper_export_step, {"artifact": str(whitepaper_export_csv)})

        if not args.skip_mteb:
            mteb_method = "rrf_rerank" if reranker is not None else "rrf"
            mteb_out_csv = args.output_dir / f"mteb_retrieved_chunks_{model_key}_{mteb_method}.csv"
            mteb_step = f"mteb_chunk__{model_key}__{mteb_method}"
            if _has_metrics("mteb_legal", "chunk", model_key, mteb_method) and _step_done(mteb_step, [mteb_out_csv]):
                print(f"[resume] Skipping existing MTEB eval: {model_key}/{mteb_method}")
                continue
            try:
                print(
                    f"[mteb] Starting chunk-level eval for model={model_key}, method={mteb_method}, "
                    f"dataset={args.mteb_dataset}, split={args.mteb_split}",
                    flush=True,
                )
                mteb_retriever = _build_mteb_retriever(
                    model_key=model_key,
                    use_reranker=(reranker is not None),
                    reranker=reranker,
                    dataset_id=args.mteb_dataset,
                    max_corpus=args.max_corpus,
                    embed_batch_size=mteb_embed_batch_size,
                    embed_device=getattr(args, "mteb_device", "auto"),
                    embed_precision=getattr(args, "mteb_precision", "float32"),
                )
                mteb_metrics = _evaluate_mteb_chunk_level(
                    retriever=mteb_retriever,
                    dataset_id=args.mteb_dataset,
                    split_name=args.mteb_split,
                    k_values=sorted(set(args.k_values)),
                    top_k=max(args.top_k * 3, 30),
                    max_corpus=args.max_corpus,
                    model_key=model_key,
                    method=mteb_method,
                    out_retrieved_csv=mteb_out_csv,
                )
                metrics_rows.extend(
                    _metrics_to_rows(
                        mteb_metrics,
                        dataset="mteb_legal",
                        level="chunk",
                        model_key=model_key,
                        method=mteb_method,
                    )
                )
                _checkpoint_metrics(f"mteb eval {model_key}/{mteb_method}")
                _write_step_checkpoint(mteb_step, {"artifact": str(mteb_out_csv)})
                print(
                    f"[mteb] Finished chunk-level eval for model={model_key}, method={mteb_method}",
                    flush=True,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if isinstance(exc, MemoryError) or "out of memory" in msg or "cuda out of memory" in msg:
                    print(f"[warn] Skipping MTEB for model={model_key} due to memory limits: {exc}")
                else:
                    print(f"[warn] Skipping MTEB for model={model_key} due to non-fatal error: {exc}")
                _write_step_checkpoint(
                    mteb_step,
                    {
                        "status": "skipped",
                        "error": str(exc),
                    },
                )
                continue

    if getattr(args, "include_splade", False):
        _run_splade_eval(
            args,
            reranker,
            metrics_rows,
            _checkpoint_metrics,
            has_metrics_fn=_has_metrics,
            step_done_fn=_step_done,
            mark_step_fn=_write_step_checkpoint,
        )

    if getattr(args, "include_colbert", False):
        _run_colbert_eval(
            args,
            reranker,
            metrics_rows,
            _checkpoint_metrics,
            has_metrics_fn=_has_metrics,
            step_done_fn=_step_done,
            mark_step_fn=_write_step_checkpoint,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_csv, index=False)

    ranking_df = (
        metrics_df[metrics_df["k"] == 10]
        .sort_values(["dataset", "level", "ndcg"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    _validate_ranking_consistency(metrics_df, ranking_df, k_for_ranking=10)
    ranking_csv = args.output_dir / "ranking_k10.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    summary_tables: list[pd.DataFrame] = []
    comparison_tables: list[pd.DataFrame] = []
    for k_val in sorted(set(args.k_values)):
        s_df, c_df = _build_metrics_summary_tables(metrics_df, k_for_summary=int(k_val))
        summary_tables.append(s_df)
        comparison_tables.append(c_df)

    summary_all_k_df = (
        pd.concat(summary_tables, ignore_index=True)
        if summary_tables
        else pd.DataFrame(columns=["dataset", "level", "model_key", "method", "k", "hit_rate", "mrr", "ndcg", "num_queries"])
    )
    comparison_all_k_df = (
        pd.concat(comparison_tables, ignore_index=True)
        if comparison_tables
        else pd.DataFrame(
            columns=[
                "dataset",
                "level",
                "method",
                "k",
                "metric",
                "best_model",
                "best_value",
                "second_model",
                "second_value",
                "gap_to_second",
                "num_models_compared",
            ]
        )
    )

    summary_k10_df = summary_all_k_df[summary_all_k_df["k"] == 10].reset_index(drop=True)
    comparison_k10_df = comparison_all_k_df[comparison_all_k_df["k"] == 10].reset_index(drop=True)

    summary_k10_csv = args.output_dir / "metrics_summary_k10.csv"
    comparison_k10_csv = args.output_dir / "comparison_k10.csv"
    summary_all_k_csv = args.output_dir / "metrics_summary_all_k.csv"
    comparison_all_k_csv = args.output_dir / "comparison_all_k.csv"
    summary_k10_df.to_csv(summary_k10_csv, index=False)
    comparison_k10_df.to_csv(comparison_k10_csv, index=False)
    summary_all_k_df.to_csv(summary_all_k_csv, index=False)
    comparison_all_k_df.to_csv(comparison_all_k_csv, index=False)

    # Flat, notebook-friendly ablation view across all requested k values.
    ablation_all_k_df = (
        metrics_df[(metrics_df["dataset"] == "gold_standard") & (metrics_df["level"] == "document")][
            ["k", "model_key", "method", "hit_rate", "mrr", "ndcg", "num_queries"]
        ]
        .copy()
        .sort_values(["k", "method", "ndcg"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    ablation_all_k_csv = args.output_dir / "ablation_table_all_k.csv"
    ablation_all_k_df.to_csv(ablation_all_k_csv, index=False)

    # Publishable ablation table (integrated from evaluation.publishable_eval)
    try:
        ablation_k = 10 if 10 in set(args.k_values) else max(set(args.k_values))
        ablation_df = build_ablation_table(metrics_csv, k=ablation_k)

        if args.with_robustness:
            per_query_csv = args.output_dir / "per_query_scores_for_ablation.csv"
            per_query_df = collect_per_query_scores(
                gold_csv=args.gold_csv,
                model_keys=args.models,
                k=ablation_k,
                skip_reranker=args.skip_reranker,
                out_csv=per_query_csv,
            )
            ablation_out = add_significance_markers(ablation_df, per_query_df)
            summary["outputs"]["per_query_scores_for_ablation"] = str(per_query_csv)
        else:
            ablation_out = ablation_df

        ablation_csv = args.output_dir / "ablation_table.csv"
        ablation_flat_csv = args.output_dir / "ablation_table_flat.csv"
        ablation_txt = args.output_dir / "ablation_table.txt"
        ablation_out.to_csv(ablation_csv)
        # Make a flat CSV variant (explicit method column) for notebooks/spreadsheets.
        ablation_flat_df = ablation_out.reset_index().rename(columns={"index": "method"})
        if isinstance(ablation_flat_df.columns, pd.MultiIndex):
            flat_cols: list[str] = []
            for col in ablation_flat_df.columns:
                parts = [str(p).strip() for p in col if str(p).strip() and str(p).strip().lower() != "nan"]
                if not parts:
                    flat_cols.append("")
                elif len(parts) == 1:
                    flat_cols.append(parts[0])
                else:
                    flat_cols.append("__".join(parts))
            ablation_flat_df.columns = flat_cols
        ablation_flat_df.to_csv(ablation_flat_csv, index=False)
        ablation_txt.write_text(
            format_ablation_report(ablation_out, k=ablation_k),
            encoding="utf-8",
        )
        summary["outputs"]["ablation_table"] = str(ablation_csv)
        summary["outputs"]["ablation_table_flat"] = str(ablation_flat_csv)
        summary["outputs"]["ablation_table_txt"] = str(ablation_txt)
    except Exception as exc:
        print(f"[warn] Ablation table generation skipped: {exc}")

    interpretation_lines: list[str] = [
        "Unified Evaluation Interpretation (k=10)",
        "",
        f"Total result rows: {len(summary_k10_df)}",
        "Top systems by dataset/level (NDCG@10):",
    ]
    top_by_group = (
        summary_k10_df.sort_values("ndcg", ascending=False)
        .groupby(["dataset", "level"], as_index=False)
        .first()
    )
    for _, row in top_by_group.iterrows():
        interpretation_lines.append(
            f"- {row['dataset']} | {row['level']}: {row['model_key']} + {row['method']} "
            f"(NDCG@10={row['ndcg']:.4f}, MRR@10={row['mrr']:.4f}, Hit@10={row['hit_rate']:.4f})"
        )
    comparable_k10 = comparison_k10_df[comparison_k10_df["num_models_compared"] >= 2].copy()
    if not comparable_k10.empty:
        interpretation_lines.append("")
        interpretation_lines.append("Largest model gaps by method:")
        biggest = comparable_k10.sort_values("gap_to_second", ascending=False).head(6)
        for _, row in biggest.iterrows():
            interpretation_lines.append(
                f"- {row['dataset']} | {row['level']} | {row['method']} | {row['metric']}: "
                f"{row['best_model']} over {row['second_model']} by {row['gap_to_second']:.4f}"
            )
    else:
        interpretation_lines.append("")
        interpretation_lines.append(
            "Model-gap section skipped: fewer than 2 models were evaluated for k=10 comparisons."
        )
    interpretation_csv = args.output_dir / "interpretation_k10.txt"
    interpretation_csv.write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    summary["outputs"]["metrics_all"] = str(metrics_csv)
    summary["outputs"]["ranking_k10"] = str(ranking_csv)
    summary["outputs"]["metrics_summary_k10"] = str(summary_k10_csv)
    summary["outputs"]["comparison_k10"] = str(comparison_k10_csv)
    summary["outputs"]["metrics_summary_all_k"] = str(summary_all_k_csv)
    summary["outputs"]["comparison_all_k"] = str(comparison_all_k_csv)
    summary["outputs"]["ablation_table_all_k"] = str(ablation_all_k_csv)
    summary["outputs"]["interpretation_k10"] = str(interpretation_csv)
    summary_json = args.output_dir / "run_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[done] Unified evaluation finished.")
    print(f"[done] Metrics: {metrics_csv}")
    print(f"[done] Ranking: {ranking_csv}")
    print(f"[done] Summary: {summary_json}")
