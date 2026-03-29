"""CSV export helpers used by unified evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import normalise_doc_name
from evaluation.evaluation import group_gold_by_query, load_gold_standard, load_whitepaper_recommendations
from evaluation.experiment_helpers import _safe_retrieve


def export_gold_retrieved_chunks(
    *,
    retriever: Any,
    model_key: str,
    method: str,
    gold_csv: Path,
    out_csv: Path,
    top_k: int,
) -> None:
    query_to_docs = group_gold_by_query(load_gold_standard(gold_csv))
    rows: list[dict[str, Any]] = []
    queries = list(query_to_docs.keys())
    for i, query in enumerate(queries, start=1):
        result = _safe_retrieve(retriever, query, top_k=top_k)
        if i % 50 == 0 or i == len(queries):
            print(f"[gold] {i}/{len(queries)} queries processed for {model_key}|{method}")
        for rank, (chunk, score) in enumerate(zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1):
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


def export_whitepaper_retrieved_chunks(
    *,
    retriever: Any,
    model_key: str,
    method: str,
    whitepaper_csv: Path,
    out_csv: Path,
    top_k: int,
) -> None:
    wp_rows = load_whitepaper_recommendations(whitepaper_csv)
    rows: list[dict[str, Any]] = []

    for i, wp in enumerate(wp_rows, start=1):
        query = (wp.get("recommendation", "") or "").strip()
        if not query:
            query = f"{wp.get('section', '')} {wp.get('subsection', '')} {wp.get('title', '')}".strip()

        result = _safe_retrieve(retriever, query, top_k=top_k)
        if i % 10 == 0 or i == len(wp_rows):
            print(f"[whitepaper] {i}/{len(wp_rows)} queries processed for {model_key}|{method}")
        for rank, (chunk, score) in enumerate(zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1):
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
