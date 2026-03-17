"""Extended CLI command handlers for heavy evaluation workflows.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import (
    DEFAULT_TOP_K,
    DEFAULT_RERANK_TOP,
    EMBEDDING_MODELS,
    INDEX_DIR,
    JUDGE_MODEL,
    LLM_MODEL,
    RERANKER_MODEL,
    SPLADE_MAX_LENGTH,
    SPLADE_MODEL,
    normalise_doc_name,
)
from data_models import Chunk
from embedding_indexing import (
    build_faiss_index,
    build_index,
    embed_texts,
    get_embed_model,
    load_indices,
    tokenize,
)
from evaluation.evaluation import (
    evaluate_retrieval,
    group_gold_by_query,
    load_gold_standard,
    load_whitepaper_recommendations,
    per_query_retrieval_scores,
)
from evaluation.metrics import bootstrap_ci, compute_retrieval_metrics, paired_permutation_test
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever as CompositeHybridRetriever
from retrieval.reranker import RerankedRetriever, Reranker
from retrieval.retrieval import HybridRetriever
from retrieval.splade_retriever import SPLADERetriever


def _safe_retrieve(retriever: Any, query: str, top_k: int):
    try:
        return retriever.retrieve(query, top_k=top_k)
    except TypeError:
        return retriever.retrieve(query)


def _load_split(dataset_id: str, config_name: str, split_name: str):
    from datasets import load_dataset

    try:
        return load_dataset(dataset_id, config_name, split=split_name)
    except Exception:
        return load_dataset(dataset_id, split=split_name)


def _build_mteb_chunks(corpus_ds, max_corpus: int | None) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    texts: list[str] = []
    for i, row in enumerate(corpus_ds):
        if max_corpus is not None and i >= max_corpus:
            break
        doc_id = str(row.get("_id", f"doc_{i}"))
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        merged = f"{title}\n{text}".strip()
        if not merged:
            continue
        chunks.append(
            Chunk(
                id=doc_id,
                document="MTEB-LegalBench",
                source_file="mteb_legalbench",
                version="v1",
                chapter="",
                article="",
                article_subtitle="",
                paragraph="",
                char_offset=0,
                text=merged,
            )
        )
        texts.append(merged)
    return chunks, texts


def _load_mteb_queries_qrels(
    dataset_id: str,
    split_name: str,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    queries_ds = _load_split(dataset_id, "queries", "queries")
    qrels_ds = _load_split(dataset_id, "qrels", split_name)

    queries: dict[str, str] = {}
    duplicate_query_ids: set[str] = set()
    for row in queries_ds:
        qid = str(row.get("_id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not qid or not text:
            continue
        if qid in queries:
            duplicate_query_ids.add(qid)
        queries[qid] = text

    if duplicate_query_ids:
        dup_preview = ", ".join(sorted(duplicate_query_ids)[:10])
        raise RuntimeError(
            "Duplicate query IDs detected in MTEB query split; cannot guarantee paired evaluation. "
            f"Examples: {dup_preview}"
        )

    relevant_by_query: dict[str, set[str]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", "")).strip()
        did = str(row.get("corpus-id", "")).strip()
        score = float(row.get("score", 0) or 0)
        if not qid or not did or score <= 0:
            continue
        relevant_by_query.setdefault(qid, set()).add(did)

    return queries, relevant_by_query


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    sorted_p = [float(p_values[i]) for i in order]
    adjusted_sorted: list[float] = [0.0] * m
    prev = 0.0
    for i, p in enumerate(sorted_p):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)
        adjusted_sorted[i] = adj
        prev = adj
    adjusted: list[float] = [0.0] * m
    for sorted_idx, original_idx in enumerate(order):
        adjusted[int(original_idx)] = adjusted_sorted[sorted_idx]
    return adjusted


def _paired_effect_size_dz(scores_a: list[float], scores_b: list[float]) -> float:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    if len(diff) < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff) / sd)


def _effect_size_label(dz: float) -> str:
    adz = abs(dz)
    if adz < 0.2:
        return "negligible"
    if adz < 0.5:
        return "small"
    if adz < 0.8:
        return "medium"
    return "large"


def _build_metrics_summary_tables(metrics_df: pd.DataFrame, k_for_summary: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = (
        metrics_df[metrics_df["k"] == k_for_summary][
            ["dataset", "level", "model_key", "method", "k", "hit_rate", "mrr", "ndcg", "num_queries"]
        ]
        .copy()
        .sort_values(["dataset", "level", "method", "ndcg"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    comp_rows: list[dict[str, Any]] = []
    for (dataset, level, method), group in summary_df.groupby(["dataset", "level", "method"], dropna=False):
        if len(group) < 2:
            continue
        for metric in ("ndcg", "mrr", "hit_rate"):
            ranked = group.sort_values(metric, ascending=False).reset_index(drop=True)
            best = ranked.iloc[0]
            second = ranked.iloc[1]
            comp_rows.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "method": method,
                    "metric": metric,
                    "best_model": str(best["model_key"]),
                    "best_value": float(best[metric]),
                    "second_model": str(second["model_key"]),
                    "second_value": float(second[metric]),
                    "gap_to_second": float(best[metric] - second[metric]),
                    "num_models_compared": int(len(ranked)),
                }
            )
    comparison_df = pd.DataFrame(comp_rows)
    return summary_df, comparison_df


def _validate_ranking_consistency(metrics_df: pd.DataFrame, ranking_df: pd.DataFrame, k_for_ranking: int = 10) -> None:
    expected = (
        metrics_df[metrics_df["k"] == k_for_ranking]
        .sort_values(["dataset", "level", "ndcg"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    if expected.shape != ranking_df.shape:
        raise RuntimeError(
            "Ranking export shape mismatch with evaluated metrics table. "
            f"expected={expected.shape}, actual={ranking_df.shape}"
        )
    expected_rows = expected.to_dict("records")
    actual_rows = ranking_df.reset_index(drop=True).to_dict("records")
    if expected_rows != actual_rows:
        raise RuntimeError("Ranking export does not match evaluated and sorted k=10 metrics.")


def _metrics_to_rows(
    metrics_by_k: dict[int, object],
    *,
    dataset: str,
    level: str,
    model_key: str,
    method: str,
) -> list[dict]:
    rows: list[dict] = []
    for k, m in sorted(metrics_by_k.items()):
        rows.append(
            {
                "dataset": dataset,
                "level": level,
                "model_key": model_key,
                "method": method,
                "k": k,
                "hit_rate": m.hit_rate,
                "recall": m.recall,
                "precision": m.precision,
                "mrr": m.mrr,
                "map": m.map_score,
                "ndcg": m.ndcg,
                "mean_rank": m.mean_rank,
                "num_queries": m.num_queries,
            }
        )
    return rows


def _indices_exist(model_key: str) -> bool:
    prefix = INDEX_DIR / model_key
    return (
        Path(str(prefix) + "_faiss.index").exists()
        and Path(str(prefix) + "_bm25.pkl").exists()
        and Path(str(prefix) + "_chunks.pkl").exists()
    )


def _build_retrievers_for_model(
    model_key: str,
    reranker: Reranker | None,
    top_k: int,
    rerank_top: int,
) -> dict[str, Any]:
    faiss_index, bm25_index, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)

    bm25 = BM25Retriever(bm25_index, chunks)
    dense = DenseRetriever(faiss_index, chunks, embed_model)
    hybrid = CompositeHybridRetriever(faiss_index, bm25_index, chunks, embed_model)

    retrievers: dict[str, Any] = {
        "bm25": bm25,
        "dense": dense,
        "rrf": hybrid,
    }
    if reranker is not None:
        retrievers["bm25_rerank"] = RerankedRetriever(
            bm25, reranker, initial_k=max(top_k * 2, 30), final_k=rerank_top
        )
        retrievers["dense_rerank"] = RerankedRetriever(
            dense, reranker, initial_k=max(top_k * 2, 30), final_k=rerank_top
        )
        retrievers["rrf_rerank"] = RerankedRetriever(
            hybrid, reranker, initial_k=max(top_k * 2, 30), final_k=rerank_top
        )
    return retrievers


def _evaluate_mteb_chunk_level(
    *,
    retriever,
    dataset_id: str,
    split_name: str,
    k_values: Iterable[int],
    top_k: int,
    max_corpus: int | None,
    model_key: str,
    method: str,
    out_retrieved_csv: Path | None,
) -> dict[int, object]:
    corpus_ds = _load_split(dataset_id, "corpus", "corpus")
    chunks, _ = _build_mteb_chunks(corpus_ds, max_corpus)
    allowed_doc_ids = {c.id for c in chunks}

    queries, relevant_by_query = _load_mteb_queries_qrels(dataset_id, split_name)
    filtered_qrels = {
        qid: {did for did in rels if did in allowed_doc_ids}
        for qid, rels in relevant_by_query.items()
    }
    filtered_qrels = {qid: rels for qid, rels in filtered_qrels.items() if rels and qid in queries}
    ordered_qids = sorted(filtered_qrels.keys())

    if not ordered_qids:
        raise RuntimeError("No valid MTEB qrels after filtering. Try --max-corpus none.")

    max_k = max(k_values)
    n_retrieve = max(max_k * 3, top_k, 30)

    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    rows: list[dict] = []

    for idx, qid in enumerate(ordered_qids, start=1):
        query = queries[qid]
        result = _safe_retrieve(retriever, query, top_k=n_retrieve)
        ranked_ids = [c.id for c in result.ranked_chunks]

        all_retrieved.append(ranked_ids)
        all_relevant.append(filtered_qrels[qid])

        if idx % 100 == 0 or idx == len(ordered_qids):
            print(f"[mteb] {idx}/{len(ordered_qids)} queries processed for {model_key}|{method}")

        if out_retrieved_csv is not None:
            for rank, (chunk, score) in enumerate(
                zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1
            ):
                rows.append(
                    {
                        "dataset": "mteb_legalbench",
                        "model_key": model_key,
                        "method": method,
                        "query_id": qid,
                        "query": query,
                        "rank": rank,
                        "score": float(score),
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                    }
                )

    if out_retrieved_csv is not None:
        out_retrieved_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_retrieved_csv, index=False)

    return {
        k: compute_retrieval_metrics(all_retrieved, all_relevant, k)
        for k in sorted(set(k_values))
    }


def _build_mteb_retriever(
    model_key: str,
    use_reranker: bool,
    reranker: Reranker | None,
    dataset_id: str,
    max_corpus: int | None,
):
    from rank_bm25 import BM25Okapi

    corpus_ds = _load_split(dataset_id, "corpus", "corpus")
    chunks, texts = _build_mteb_chunks(corpus_ds, max_corpus)

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    embed_model = get_embed_model(model_key)
    embeddings = embed_texts(texts, embed_model, show_progress=False)
    faiss_index = build_faiss_index(embeddings)
    hybrid = HybridRetriever(faiss_index, bm25, chunks, embed_model)

    if use_reranker and reranker is not None:
        return RerankedRetriever(
            hybrid,
            reranker,
            initial_k=max(DEFAULT_TOP_K * 2, 30),
            final_k=DEFAULT_RERANK_TOP,
        )
    return hybrid


def _build_mteb_splade_retriever(
    *,
    dataset_id: str,
    max_corpus: int | None,
    model_name: str,
    max_length: int,
):
    corpus_ds = _load_split(dataset_id, "corpus", "corpus")
    chunks, _ = _build_mteb_chunks(corpus_ds, max_corpus)
    return SPLADERetriever.from_chunks(
        chunks,
        model_name=model_name,
        max_length=max_length,
    )


def _export_gold_retrieved_chunks(
    *,
    retriever,
    model_key: str,
    method: str,
    gold_csv: Path,
    out_csv: Path,
    top_k: int,
) -> None:
    query_to_docs = group_gold_by_query(load_gold_standard(gold_csv))
    rows: list[dict] = []
    queries = list(query_to_docs.keys())
    for i, query in enumerate(queries, start=1):
        result = _safe_retrieve(retriever, query, top_k=top_k)
        if i % 50 == 0 or i == len(queries):
            print(f"[gold] {i}/{len(queries)} queries processed for {model_key}|{method}")
        for rank, (chunk, score) in enumerate(
            zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1
        ):
            rows.append(
                {
                    "dataset": "gold_standard",
                    "model_key": model_key,
                    "method": method,
                    "query": query,
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.id,
                    "document": chunk.document,
                    "document_canonical": normalise_doc_name(chunk.document),
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                    "text": chunk.text,
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _export_whitepaper_retrieved_chunks(
    *,
    retriever,
    model_key: str,
    method: str,
    whitepaper_csv: Path,
    out_csv: Path,
    top_k: int,
) -> None:
    wp_rows = load_whitepaper_recommendations(whitepaper_csv)
    rows: list[dict] = []

    for i, wp in enumerate(wp_rows, start=1):
        query = (wp.get("recommendation", "") or "").strip()
        if not query:
            query = f"{wp.get('section', '')} {wp.get('subsection', '')} {wp.get('title', '')}".strip()

        result = _safe_retrieve(retriever, query, top_k=top_k)
        if i % 10 == 0 or i == len(wp_rows):
            print(f"[whitepaper] {i}/{len(wp_rows)} queries processed for {model_key}|{method}")
        for rank, (chunk, score) in enumerate(
            zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1
        ):
            rows.append(
                {
                    "dataset": "whitepaper",
                    "model_key": model_key,
                    "method": method,
                    "section": wp.get("section", ""),
                    "subsection": wp.get("subsection", ""),
                    "title": wp.get("title", ""),
                    "query": query,
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.id,
                    "document": chunk.document,
                    "document_canonical": normalise_doc_name(chunk.document),
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                    "text": chunk.text,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _mean_or_inf(arr: list[float]) -> float:
    finite = [x for x in arr if x != float("inf")]
    return float(np.mean(finite)) if finite else float("inf")


def _error_categories(
    retriever: Any,
    gold_csv: Path,
    k_main: int,
    top_k: int,
    rerank_top: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = load_gold_standard(gold_csv)
    query_to_docs = group_gold_by_query(entries)

    rows: list[dict[str, Any]] = []
    max_probe = max(20, top_k)

    for query, relevant in query_to_docs.items():
        try:
            result = retriever.retrieve(query, top_k=max_probe * 3, rerank_top=max_probe)
        except TypeError:
            result = retriever.retrieve(query, top_k=max_probe)

        seen = []
        seen_set = set()
        for chunk in result.ranked_chunks:
            doc = chunk.document
            if doc not in seen_set:
                seen.append(doc)
                seen_set.add(doc)

        topk = set(seen[:k_main])
        top20 = set(seen[:20])
        rel = set(relevant)

        n_topk = len(rel & topk)
        if n_topk == len(rel):
            category = "success"
        elif n_topk > 0:
            category = "partial"
        elif rel & top20:
            category = "late_hit"
        else:
            category = "complete_miss"

        rows.append(
            {
                "query": query,
                "relevant_docs": "; ".join(sorted(rel)),
                "retrieved_top10_docs": "; ".join(seen[:10]),
                "category": category,
            }
        )

    per_query_df = pd.DataFrame(rows)
    counts_df = (
        per_query_df["category"].value_counts(dropna=False).rename_axis("category").reset_index(name="count")
    )
    counts_df["share"] = counts_df["count"] / max(len(per_query_df), 1)
    return per_query_df, counts_df


def cmd_download_models(args: argparse.Namespace) -> None:
    """Pre-download embedding/reranker/LLM models."""
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for key in args.embedding_models:
        if key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model key: {key}")
        model_name = EMBEDDING_MODELS[key]
        print(f"[download] SentenceTransformer: {model_name}")
        SentenceTransformer(model_name)

    print(f"[download] CrossEncoder: {RERANKER_MODEL}")
    CrossEncoder(RERANKER_MODEL)

    if args.include_llms:
        for model_name in (LLM_MODEL, JUDGE_MODEL):
            print(f"[download] Tokenizer: {model_name}")
            AutoTokenizer.from_pretrained(model_name)
            print(f"[download] CausalLM: {model_name}")
            AutoModelForCausalLM.from_pretrained(model_name)

    print("[download] Completed.")


def cmd_unified_eval(args: argparse.Namespace) -> None:
    """Run unified evaluation across gold, MTEB, and whitepaper exports."""
    if args.full_mteb:
        args.max_corpus = None
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    summary: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "k_values": sorted(set(args.k_values)),
        "mteb_dataset": args.mteb_dataset,
        "mteb_split": args.mteb_split,
        "mteb_max_corpus": args.max_corpus,
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
        "outputs": {},
    }

    reranker = None
    if not args.skip_reranker:
        print("[setup] Loading cross-encoder reranker...")
        reranker = Reranker()

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

        retrievers = _build_retrievers_for_model(
            model_key,
            reranker=reranker,
            top_k=args.top_k,
            rerank_top=args.rerank_top,
        )

        for method_name, retriever in retrievers.items():
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

        export_method = "rrf_rerank" if "rrf_rerank" in retrievers else "rrf"
        _export_gold_retrieved_chunks(
            retriever=retrievers[export_method],
            model_key=model_key,
            method=export_method,
            gold_csv=args.gold_csv,
            out_csv=args.output_dir / f"gold_retrieved_chunks_{model_key}_{export_method}.csv",
            top_k=args.export_k,
        )

        if not args.skip_whitepaper:
            _export_whitepaper_retrieved_chunks(
                retriever=retrievers[export_method],
                model_key=model_key,
                method=export_method,
                whitepaper_csv=args.whitepaper_csv,
                out_csv=args.output_dir / f"whitepaper_retrieved_chunks_{model_key}_{export_method}.csv",
                top_k=args.export_k,
            )

        if not args.skip_mteb:
            mteb_method = "rrf_rerank" if reranker is not None else "rrf"
            mteb_retriever = _build_mteb_retriever(
                model_key=model_key,
                use_reranker=(reranker is not None),
                reranker=reranker,
                dataset_id=args.mteb_dataset,
                max_corpus=args.max_corpus,
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
                out_retrieved_csv=args.output_dir / f"mteb_retrieved_chunks_{model_key}_{mteb_method}.csv",
            )
            metrics_rows.extend(
                _metrics_to_rows(
                    mteb_metrics,
                    dataset="mteb_legalbench",
                    level="chunk",
                    model_key=model_key,
                    method=mteb_method,
                )
            )

    if getattr(args, "include_splade", False):
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

        export_method = "splade_rerank" if "splade_rerank" in splade_retrievers else "splade"
        _export_gold_retrieved_chunks(
            retriever=splade_retrievers[export_method],
            model_key="splade",
            method=export_method,
            gold_csv=args.gold_csv,
            out_csv=args.output_dir / f"gold_retrieved_chunks_splade_{export_method}.csv",
            top_k=args.export_k,
        )

        if not args.skip_whitepaper:
            _export_whitepaper_retrieved_chunks(
                retriever=splade_retrievers[export_method],
                model_key="splade",
                method=export_method,
                whitepaper_csv=args.whitepaper_csv,
                out_csv=args.output_dir / f"whitepaper_retrieved_chunks_splade_{export_method}.csv",
                top_k=args.export_k,
            )

        if not args.skip_mteb:
            mteb_method = "splade_rerank" if reranker is not None else "splade"
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

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = args.output_dir / "metrics_all.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    ranking_df = (
        metrics_df[metrics_df["k"] == 10]
        .sort_values(["dataset", "level", "ndcg"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    _validate_ranking_consistency(metrics_df, ranking_df, k_for_ranking=10)
    ranking_csv = args.output_dir / "ranking_k10.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    summary_k10_df, comparison_k10_df = _build_metrics_summary_tables(metrics_df, k_for_summary=10)
    summary_k10_csv = args.output_dir / "metrics_summary_k10.csv"
    comparison_k10_csv = args.output_dir / "comparison_k10.csv"
    summary_k10_df.to_csv(summary_k10_csv, index=False)
    comparison_k10_df.to_csv(comparison_k10_csv, index=False)

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
    if not comparison_k10_df.empty:
        interpretation_lines.append("")
        interpretation_lines.append("Largest model gaps by method:")
        biggest = comparison_k10_df.sort_values("gap_to_second", ascending=False).head(6)
        for _, row in biggest.iterrows():
            interpretation_lines.append(
                f"- {row['dataset']} | {row['level']} | {row['method']} | {row['metric']}: "
                f"{row['best_model']} over {row['second_model']} by {row['gap_to_second']:.4f}"
            )
    interpretation_csv = args.output_dir / "interpretation_k10.txt"
    interpretation_csv.write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    summary["outputs"]["metrics_all"] = str(metrics_csv)
    summary["outputs"]["ranking_k10"] = str(ranking_csv)
    summary["outputs"]["metrics_summary_k10"] = str(summary_k10_csv)
    summary["outputs"]["comparison_k10"] = str(comparison_k10_csv)
    summary["outputs"]["interpretation_k10"] = str(interpretation_csv)
    summary_json = args.output_dir / "run_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[done] Unified evaluation finished.")
    print(f"[done] Metrics: {metrics_csv}")
    print(f"[done] Ranking: {ranking_csv}")
    print(f"[done] Summary: {summary_json}")


def cmd_robustness(args: argparse.Namespace) -> None:
    """Run publishability-focused robustness analysis."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    idx_prefix = INDEX_DIR / args.model
    if not (Path(str(idx_prefix) + "_faiss.index").exists() and Path(str(idx_prefix) + "_bm25.pkl").exists()):
        raise FileNotFoundError(
            f"Indices for model '{args.model}' are missing. Build first with: "
            f"python main.py build -i outputs/evidence.csv -m {args.model}"
        )

    retrievers = _build_retrievers_for_model(
        args.model,
        top_k=args.top_k,
        rerank_top=args.rerank_top,
        reranker=None if args.skip_reranker else Reranker(),
    )

    gold_queries = sorted(group_gold_by_query(load_gold_standard(args.gold_csv)).keys())

    per_query_long_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []

    for method_name, retriever in retrievers.items():
        scores = per_query_retrieval_scores(
            retriever,
            gold_path=args.gold_csv,
            k=args.k,
            top_k_retrieve=max(args.top_k * 3, 30),
            rerank_top=args.rerank_top,
            level="document",
        )

        n = len(scores["hit"])
        if n != len(gold_queries):
            raise RuntimeError(
                "Per-query score length mismatch with gold query set: "
                f"scores={n}, gold_queries={len(gold_queries)}"
            )
        for i in range(n):
            per_query_long_rows.append(
                {
                    "model_key": args.model,
                    "method": method_name,
                    "query_index": i,
                    "query": gold_queries[i],
                    "hit": float(scores["hit"][i]),
                    "recall": float(scores["recall"][i]),
                    "precision": float(scores["precision"][i]),
                    "mrr": float(scores["mrr"][i]),
                    "ap": float(scores["ap"][i]),
                    "ndcg": float(scores["ndcg"][i]),
                    "rank": float(scores["rank"][i]),
                }
            )

        for metric in ("hit", "mrr", "ndcg"):
            mean_val, ci_lo, ci_hi = bootstrap_ci(scores[metric], n_bootstrap=10_000, confidence=0.95, rng_seed=42)
            ci_rows.append(
                {
                    "model_key": args.model,
                    "method": method_name,
                    "k": args.k,
                    "metric": metric,
                    "mean": mean_val,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                    "num_queries": len(scores[metric]),
                }
            )

        ci_rows.append(
            {
                "model_key": args.model,
                "method": method_name,
                "k": args.k,
                "metric": "mean_rank",
                "mean": _mean_or_inf(scores["rank"]),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "num_queries": len(scores["rank"]),
            }
        )

    per_query_df = pd.DataFrame(per_query_long_rows)
    per_query_csv = args.output_dir / f"per_query_scores_k{args.k}_{args.model}.csv"
    per_query_df.to_csv(per_query_csv, index=False)

    ci_df = pd.DataFrame(ci_rows)
    ci_csv = args.output_dir / f"ci_table_k{args.k}_{args.model}.csv"
    ci_df.to_csv(ci_csv, index=False)

    pair_rows: list[dict[str, Any]] = []
    methods = sorted(per_query_df["method"].unique().tolist())
    for a, b in combinations(methods, 2):
        dfa = per_query_df[per_query_df["method"] == a].sort_values("query_index").reset_index(drop=True)
        dfb = per_query_df[per_query_df["method"] == b].sort_values("query_index").reset_index(drop=True)
        if not dfa["query_index"].equals(dfb["query_index"]):
            raise RuntimeError(f"Query index mismatch for paired comparison: {a} vs {b}")
        if not dfa["query"].equals(dfb["query"]):
            raise RuntimeError(f"Query text mismatch for paired comparison: {a} vs {b}")
        for metric in ("mrr", "ndcg", "hit"):
            scores_a = dfa[metric].astype(float).tolist()
            scores_b = dfb[metric].astype(float).tolist()
            p_val = paired_permutation_test(
                scores_a,
                scores_b,
                n_permutations=10_000,
                rng_seed=42,
            )
            delta = float(dfa[metric].mean() - dfb[metric].mean())
            effect_dz = _paired_effect_size_dz(scores_a, scores_b)
            pair_rows.append(
                {
                    "model_key": args.model,
                    "k": args.k,
                    "metric": metric,
                    "method_a": a,
                    "method_b": b,
                    "delta_mean": delta,
                    "p_value": p_val,
                    "effect_size_dz": effect_dz,
                    "effect_label": _effect_size_label(effect_dz),
                }
            )

    pvals_df = pd.DataFrame(pair_rows)
    if not pvals_df.empty:
        pvals_df["p_value_holm"] = np.nan
        pvals_df["significant_holm_0_05"] = False
        for metric, group in pvals_df.groupby("metric", sort=False):
            adjusted = _holm_bonferroni(group["p_value"].astype(float).tolist())
            pvals_df.loc[group.index, "p_value_holm"] = adjusted
            pvals_df.loc[group.index, "significant_holm_0_05"] = [float(x) < 0.05 for x in adjusted]
    pvals_csv = args.output_dir / f"pairwise_permutation_k{args.k}_{args.model}.csv"
    pvals_df.to_csv(pvals_csv, index=False)

    comparison_report_csv = args.output_dir / f"comparison_report_k{args.k}_{args.model}.csv"
    comparison_report_df = (
        pvals_df.sort_values(["metric", "p_value_holm", "p_value", "delta_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    comparison_report_df.to_csv(comparison_report_csv, index=False)

    mean_by_method_metric = (
        per_query_df.groupby("method")[["hit", "mrr", "ndcg", "rank"]].mean().to_dict("index")
    )
    ablations = [
        ("bm25", "bm25_rerank", "reranker_gain_bm25"),
        ("dense", "dense_rerank", "reranker_gain_dense"),
        ("rrf", "rrf_rerank", "reranker_gain_rrf"),
        ("dense", "rrf", "hybrid_vs_dense"),
        ("bm25", "rrf", "hybrid_vs_bm25"),
    ]
    abl_rows: list[dict[str, Any]] = []
    for base, variant, name in ablations:
        if base not in mean_by_method_metric or variant not in mean_by_method_metric:
            continue
        b = mean_by_method_metric[base]
        v = mean_by_method_metric[variant]
        abl_rows.append(
            {
                "model_key": args.model,
                "ablation": name,
                "base": base,
                "variant": variant,
                "delta_hit": float(v["hit"] - b["hit"]),
                "delta_mrr": float(v["mrr"] - b["mrr"]),
                "delta_ndcg": float(v["ndcg"] - b["ndcg"]),
                "delta_mean_rank": float(v["rank"] - b["rank"]),
            }
        )

    ablation_df = pd.DataFrame(abl_rows)
    ablation_csv = args.output_dir / f"ablation_deltas_k{args.k}_{args.model}.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    ndcg_means = per_query_df.groupby("method")["ndcg"].mean().sort_values(ascending=False)
    best_method = ndcg_means.index[0]
    best_retriever = retrievers[best_method]
    per_query_error_df, error_counts_df = _error_categories(
        best_retriever,
        args.gold_csv,
        k_main=args.k,
        top_k=args.top_k,
        rerank_top=args.rerank_top,
    )

    errors_csv = args.output_dir / f"error_analysis_queries_k{args.k}_{args.model}_{best_method}.csv"
    errors_counts_csv = args.output_dir / f"error_analysis_summary_k{args.k}_{args.model}_{best_method}.csv"
    per_query_error_df.to_csv(errors_csv, index=False)
    error_counts_df.to_csv(errors_counts_csv, index=False)

    negative_csv = args.output_dir / f"negative_cases_k{args.k}_{args.model}_{best_method}.csv"
    per_query_error_df[per_query_error_df["category"].isin(["late_hit", "complete_miss"])].to_csv(
        negative_csv, index=False
    )

    interpretation_txt = args.output_dir / f"robustness_interpretation_k{args.k}_{args.model}.txt"
    interp_lines = [
        f"Robustness interpretation for model={args.model}, k={args.k}",
        "",
        "Method means (hit, mrr, ndcg, rank):",
    ]
    means_table = per_query_df.groupby("method")[["hit", "mrr", "ndcg", "rank"]].mean().sort_values("ndcg", ascending=False)
    for method_name, row in means_table.iterrows():
        interp_lines.append(
            f"- {method_name}: hit={row['hit']:.4f}, mrr={row['mrr']:.4f}, ndcg={row['ndcg']:.4f}, rank={row['rank']:.2f}"
        )
    if not comparison_report_df.empty:
        interp_lines.append("")
        interp_lines.append("Paired differences significant after Holm correction (alpha=0.05):")
        sig_df = comparison_report_df[comparison_report_df["significant_holm_0_05"]]
        if sig_df.empty:
            interp_lines.append("- None")
        else:
            for _, row in sig_df.iterrows():
                interp_lines.append(
                    f"- {row['metric']}: {row['method_a']} vs {row['method_b']} "
                    f"(delta={row['delta_mean']:.4f}, p_holm={row['p_value_holm']:.4g}, effect={row['effect_label']}, dz={row['effect_size_dz']:.3f})"
                )
    interpretation_txt.write_text("\n".join(interp_lines) + "\n", encoding="utf-8")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "k": args.k,
        "best_method_by_ndcg": best_method,
        "outputs": {
            "per_query_scores": str(per_query_csv),
            "ci_table": str(ci_csv),
            "pairwise_permutation": str(pvals_csv),
            "comparison_report": str(comparison_report_csv),
            "ablation_deltas": str(ablation_csv),
            "error_queries": str(errors_csv),
            "error_summary": str(errors_counts_csv),
            "negative_cases": str(negative_csv),
            "interpretation": str(interpretation_txt),
        },
    }
    with open(args.output_dir / f"robustness_summary_k{args.k}_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done] robustness analysis complete")
