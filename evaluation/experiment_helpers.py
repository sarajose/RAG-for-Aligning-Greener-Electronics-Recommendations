"""Shared helpers for unified and robustness evaluation commands."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Any, Iterable

from config import DEFAULT_RERANK_TOP, DEFAULT_TOP_K, INDEX_DIR, RRF_K
from data_models import Chunk
from embedding_indexing import build_faiss_index, embed_texts, get_embed_model, load_indices, tokenize
from evaluation.metrics import compute_retrieval_metrics
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever as CompositeHybridRetriever
from retrieval.reranker import RerankedRetriever, Reranker
from retrieval.retrieval import HybridRetriever
from retrieval.splade_retriever import SPLADERetriever


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(message: str) -> None:
    print(f"[{_ts()}] [mteb] {message}", flush=True)

def _safe_retrieve(retriever: Any, query: str, top_k: int):
    try:
        return retriever.retrieve(query, top_k=top_k)
    except TypeError:
        return retriever.retrieve(query)


def _load_split(dataset_id: str, config_name: str, split_name: str):
    from datasets import load_dataset, load_from_disk

    def _local_candidates(root: Path, name: str):
        seen: set[str] = set()
        config_aliases = [name]
        if name == "qrels":
            # HF cache for this dataset commonly stores qrels under "default".
            config_aliases.append("default")

        for alias in config_aliases + [""]:
            base = root / alias if alias else root
            if not base.exists() or not base.is_dir():
                continue
            key = str(base.resolve())
            if key not in seen:
                seen.add(key)
                yield base

            # Support cache layout: <base>/<version>/<hash>
            for version_dir in base.iterdir():
                if not version_dir.is_dir():
                    continue
                for hash_dir in version_dir.iterdir():
                    if not hash_dir.is_dir():
                        continue
                    hash_key = str(hash_dir.resolve())
                    if hash_key in seen:
                        continue
                    seen.add(hash_key)
                    yield hash_dir

    ds_path = Path(dataset_id)
    if ds_path.exists() and ds_path.is_dir():
        # Support local layouts such as:
        # 1) <root>/<config_name>/... saved with load_from_disk
        # 2) <root>/... as a DatasetDict or Dataset
        for candidate in _local_candidates(ds_path, config_name):
            try:
                local_obj = load_from_disk(str(candidate))
            except Exception:
                continue

            if hasattr(local_obj, "column_names"):
                return local_obj

            if split_name in local_obj:
                return local_obj[split_name]
            if config_name in local_obj:
                return local_obj[config_name]
            if "train" in local_obj:
                return local_obj["train"]
            first_key = next(iter(local_obj.keys()), None)
            if first_key is not None:
                return local_obj[first_key]

        raise RuntimeError(
            f"Could not load local dataset split '{split_name}' from '{dataset_id}'. "
            "Expected either a Dataset/DatasetDict at the root or subfolders named corpus/queries/qrels."
        )

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
        for metric in ("ndcg", "mrr", "hit_rate"):
            ranked = group.sort_values(metric, ascending=False).reset_index(drop=True)
            best = ranked.iloc[0]
            if len(ranked) >= 2:
                second_model = str(ranked.iloc[1]["model_key"])
                second_value = float(ranked.iloc[1][metric])
                gap = float(best[metric] - ranked.iloc[1][metric])
            else:
                second_model = ""
                second_value = float("nan")
                gap = float("nan")
            comp_rows.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "method": method,
                    "k": int(k_for_summary),
                    "metric": metric,
                    "best_model": str(best["model_key"]),
                    "best_value": float(best[metric]),
                    "second_model": second_model,
                    "second_value": second_value,
                    "gap_to_second": gap,
                    "num_models_compared": int(len(ranked)),
                }
            )
    comparison_df = pd.DataFrame(
        comp_rows,
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
        ],
    )
    return summary_df, comparison_df


def _validate_ranking_consistency(metrics_df: pd.DataFrame, ranking_df: pd.DataFrame, k_for_ranking: int = 10) -> None:
    expected = (
        metrics_df[metrics_df["k"] == k_for_ranking]
        .sort_values(
            ["dataset", "level", "ndcg", "model_key", "method"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    actual = (
        ranking_df.copy()
        .sort_values(
            ["dataset", "level", "ndcg", "model_key", "method"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if expected.shape != ranking_df.shape:
        raise RuntimeError(
            "Ranking export shape mismatch with evaluated metrics table. "
            f"expected={expected.shape}, actual={ranking_df.shape}"
        )
    if list(expected.columns) != list(actual.columns):
        raise RuntimeError(
            "Ranking export columns mismatch with evaluated metrics table. "
            f"expected={list(expected.columns)}, actual={list(actual.columns)}"
        )

    try:
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "Ranking export does not match evaluated and sorted k=10 metrics. "
            f"Details: {exc}"
        ) from exc


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
    rrf_k: int = RRF_K,
) -> dict[str, Any]:
    faiss_index, bm25_index, chunks = load_indices(model_key)
    embed_model = get_embed_model(model_key)

    bm25 = BM25Retriever(bm25_index, chunks)
    dense = DenseRetriever(faiss_index, chunks, embed_model)
    hybrid = CompositeHybridRetriever(faiss_index, bm25_index, chunks, embed_model, rrf_k=rrf_k)

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
    start_ts = time.perf_counter()
    _log_progress(
        f"START eval: model={model_key}, method={method}, dataset={dataset_id}, split={split_name}"
    )
    _log_progress("Loading corpus split...")
    corpus_ds = _load_split(dataset_id, "corpus", "corpus")
    _log_progress("Building in-memory chunks from corpus...")
    chunks, _ = _build_mteb_chunks(corpus_ds, max_corpus)
    _log_progress(f"Corpus prepared with {len(chunks)} chunks.")
    allowed_doc_ids = {c.id for c in chunks}

    _log_progress("Loading query and qrels splits...")
    queries, relevant_by_query = _load_mteb_queries_qrels(dataset_id, split_name)
    filtered_qrels = {
        qid: {did for did in rels if did in allowed_doc_ids}
        for qid, rels in relevant_by_query.items()
    }
    filtered_qrels = {qid: rels for qid, rels in filtered_qrels.items() if rels and qid in queries}
    ordered_qids = sorted(filtered_qrels.keys())
    _log_progress(
        f"Prepared {len(ordered_qids)} valid queries with positive qrels (from {len(queries)} total queries)."
    )

    if not ordered_qids:
        raise RuntimeError("No valid MTEB qrels after filtering. Try --max-corpus none.")

    max_k = max(k_values)
    n_retrieve = max(max_k * 3, top_k, 30)

    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    rows: list[dict] = []
    loop_start_ts = time.perf_counter()
    heartbeat_every = 25

    for idx, qid in enumerate(ordered_qids, start=1):
        query = queries[qid]
        result = _safe_retrieve(retriever, query, top_k=n_retrieve)
        ranked_ids = [c.id for c in result.ranked_chunks]

        all_retrieved.append(ranked_ids)
        all_relevant.append(filtered_qrels[qid])

        if idx % heartbeat_every == 0 or idx == len(ordered_qids):
            elapsed = time.perf_counter() - loop_start_ts
            qps = idx / elapsed if elapsed > 0 else 0.0
            remaining = len(ordered_qids) - idx
            eta_s = int(remaining / qps) if qps > 0 else -1
            eta_txt = "unknown" if eta_s < 0 else f"~{eta_s}s"
            _log_progress(
                f"{idx}/{len(ordered_qids)} queries processed for {model_key}|{method} "
                f"({qps:.2f} q/s, ETA {eta_txt})"
            )

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
        _log_progress(f"Writing retrieved chunks CSV -> {out_retrieved_csv}")
        pd.DataFrame(rows).to_csv(out_retrieved_csv, index=False)

    total_elapsed = time.perf_counter() - start_ts
    _log_progress(
        f"DONE eval: model={model_key}, method={method}, queries={len(ordered_qids)}, elapsed={total_elapsed:.1f}s"
    )

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
    embed_batch_size: int = 32,
):
    from rank_bm25 import BM25Okapi

    _log_progress(f"Preparing hybrid retriever for model={model_key}, dataset={dataset_id}")
    _log_progress("Loading corpus split for retriever build...")
    corpus_ds = _load_split(dataset_id, "corpus", "corpus")
    _log_progress("Building MTEB chunks/texts for retriever build...")
    chunks, texts = _build_mteb_chunks(corpus_ds, max_corpus)
    _log_progress(f"Tokenizing and building BM25 over {len(texts)} texts...")

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    _log_progress(f"Loading embedding model {model_key}...")
    embed_model = get_embed_model(model_key)
    _log_progress(
        f"Embedding {len(texts)} corpus texts (this can take a while) "
        f"with batch_size={embed_batch_size}..."
    )
    embeddings = embed_texts(texts, embed_model, batch_size=embed_batch_size, show_progress=True)
    _log_progress("Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)
    hybrid = HybridRetriever(faiss_index, bm25, chunks, embed_model)
    _log_progress("Hybrid retriever is ready.")

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
    _log_progress(f"Preparing SPLADE retriever for dataset={dataset_id}, model={model_name}")
    _log_progress("Loading corpus split for SPLADE retriever...")
    corpus_ds = _load_split(dataset_id, "corpus", "corpus")
    _log_progress("Building MTEB chunks for SPLADE retriever...")
    chunks, _ = _build_mteb_chunks(corpus_ds, max_corpus)
    _log_progress(f"Building SPLADE index for {len(chunks)} chunks (this can take a while)...")
    return SPLADERetriever.from_chunks(
        chunks,
        model_name=model_name,
        max_length=max_length,
    )
