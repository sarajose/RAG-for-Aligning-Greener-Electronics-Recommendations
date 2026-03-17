from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from data_models import ClassificationResult, Recommendation


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return dialect.delimiter
    except csv.Error:
        return ","


def load_recommendations(csv_path: Path) -> list[Recommendation]:
    """Load recommendation rows from comma or semicolon CSV."""
    recs: list[Recommendation] = []
    with open(csv_path, encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=_detect_delimiter(sample))

        for row in reader:
            section = (row.get("section") or "").strip()
            subsection = (row.get("subsection") or "").strip()
            title = (row.get("title") or "").strip()
            recommendation = (row.get("recommendation") or row.get("text") or "").strip()
            text = recommendation or f"{section} {subsection} {title}".strip()
            recs.append(
                Recommendation(
                    section=section,
                    subsection=subsection,
                    title=title,
                    text=text,
                )
            )

    return recs


def save_retrieved_chunks_csv(
    queries: list[str],
    retrieval_results: list[Any],
    output_path: Path,
    top_k: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for query, result in zip(queries, retrieval_results):
        for rank, (chunk, score) in enumerate(
            zip(result.ranked_chunks[:top_k], result.scores[:top_k]), start=1
        ):
            rows.append(
                {
                    "query": query,
                    "rank": rank,
                    "chunk_id": chunk.id,
                    "document": chunk.document,
                    "article": chunk.article,
                    "paragraph": chunk.paragraph,
                    "score": float(score),
                    "text": chunk.text,
                }
            )

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_judge_results_csv(judge_results: list[Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "recommendation": r.recommendation,
            "predicted_label": r.predicted_label,
            "label_score": r.label_score,
            "justification_score": r.justification_score,
            "evidence_score": r.evidence_score,
            "overall_score": r.overall_score,
            "reasoning": r.reasoning,
        }
        for r in judge_results
    ]

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_prompt_output_csv(
    output_path: Path,
    recs: list[Recommendation],
    retrieval_results: list[Any],
    cls_results: list[ClassificationResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for idx, rec in enumerate(recs):
        ret = retrieval_results[idx]
        cls = cls_results[idx] if idx < len(cls_results) else None
        rows.append(
            {
                "section": rec.section,
                "subsection": rec.subsection,
                "title": rec.title,
                "recommendation_query": rec.text,
                "retrieved_documents": "; ".join(sorted({c.document for c in ret.ranked_chunks})),
                "retrieved_articles": "; ".join(sorted({c.article for c in ret.ranked_chunks if c.article})),
                "top_chunk_ids": "; ".join(c.id for c in ret.ranked_chunks),
                "top_chunk_texts": "\n---\n".join(
                    f"[{c.document} | {c.article}]\n{c.text[:400]}" for c in ret.ranked_chunks
                ),
                "alignment_label": cls.label if cls else "",
                "justification": cls.justification if cls else "",
                "cited_chunk_ids": "; ".join(cls.cited_chunk_ids) if cls else "",
                "human_label": "",
                "human_notes": "",
            }
        )

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
