from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path

from data_models import Chunk


def _generate_chunk_id(document: str, article: str, paragraph: str, text: str) -> str:
    """Create a stable hash-based chunk id when the CSV lacks one."""
    base = f"{document}|{article}|{paragraph}"
    txt_hash = sha256(text.encode()).hexdigest()[:8]
    return f"{base}|{txt_hash}"


def load_chunks(csv_path: Path) -> list[Chunk]:
    """Read chunked evidence from a CSV file.

    Handles both full evidence.csv and reduced evidence_recommendation.csv by
    filling missing fields with defaults.
    """
    chunks: list[Chunk] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "")
            chunk_id = row.get("id", "") or _generate_chunk_id(
                row.get("document", ""),
                row.get("article", ""),
                row.get("paragraph", ""),
                text,
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document=row.get("document", ""),
                    source_file=row.get("source_file", ""),
                    version=row.get("version", ""),
                    chapter=row.get("chapter", ""),
                    article=row.get("article", ""),
                    article_subtitle=row.get("article_subtitle", ""),
                    paragraph=row.get("paragraph", ""),
                    char_offset=int(row["char_offset"]) if row.get("char_offset") else 0,
                    text=text,
                    article_text=row.get("article_text", ""),
                )
            )
    return chunks


def load_and_merge_chunks(*csv_paths: Path) -> list[Chunk]:
    """Load and merge chunks from one or more CSV files."""
    seen_ids: set[str] = set()
    merged: list[Chunk] = []
    for path in csv_paths:
        for chunk in load_chunks(path):
            if chunk.id in seen_ids:
                continue
            seen_ids.add(chunk.id)
            merged.append(chunk)
    return merged
