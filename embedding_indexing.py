"""
Embedding generation and vector-index construction.

Builds dense-vector (FAISS HNSW) and sparse-lexical (BM25Okapi) indices
from chunked evidence CSVs.  Higher-level search orchestration lives in
:pymod:`retrieval`.

Supports multiple embedding models (see :pydata:`config.EMBEDDING_MODELS`).

Usage (standalone index build)::

    python embedding_indexing.py -i outputs/evidence.csv -m bge-m3
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
import pickle
from hashlib import sha256
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODELS,
    DEFAULT_MODEL_KEY,
    INDEX_DIR,
    FAISS_HNSW_M,
    FAISS_EF_CONSTRUCT,
    FAISS_EF_SEARCH,
)

# Re-export Chunk so that pickled indices created with the old code (which
# stored ``embedding_indexing.Chunk`` objects) can still be loaded.
from data_models import Chunk


# Data loading

def _generate_chunk_id(document: str, article: str, paragraph: str,
                       text: str) -> str:
    """Create a stable hash-based ID when the CSV lacks one."""
    base = f"{document}|{article}|{paragraph}"
    txt_hash = sha256(text.encode()).hexdigest()[:8]
    return f"{base}|{txt_hash}"


def load_chunks(csv_path: Path) -> list[Chunk]:
    """Read chunked evidence from a CSV file.

    Handles both the full-column ``evidence.csv`` and the reduced-column
    ``evidence_recommendation.csv`` by filling missing fields with
    sensible defaults.

    Parameters
    ----------
    csv_path : Path
        CSV with at least ``document, article, text`` columns.

    Returns
    -------
    list[Chunk]
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
            chunks.append(Chunk(
                id=chunk_id,
                document=row.get("document", ""),
                source_file=row.get("source_file", ""),
                version=row.get("version", ""),
                chapter=row.get("chapter", ""),
                article=row.get("article", ""),
                article_subtitle=row.get("article_subtitle", ""),
                paragraph=row.get("paragraph", ""),
                char_offset=int(row["char_offset"])
                    if row.get("char_offset") else 0,
                text=text,
            ))
    return chunks


def load_and_merge_chunks(*csv_paths: Path) -> list[Chunk]:
    """Load and merge chunks from one or more CSV files.

    Useful for combining ``evidence.csv`` (core legislation) with
    ``evidence_recommendation.csv`` (strategy / framework documents).

    Parameters
    ----------
    *csv_paths : Path
        One or more CSV file paths.

    Returns
    -------
    list[Chunk]
        De-duplicated by chunk ID.
    """
    seen_ids: set[str] = set()
    merged: list[Chunk] = []
    for path in csv_paths:
        for chunk in load_chunks(path):
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                merged.append(chunk)
    return merged



# Embedding

def get_embed_model(model_key: str = DEFAULT_MODEL_KEY) -> SentenceTransformer:
    """Load a sentence-transformer embedding model.

    Parameters
    ----------
    model_key : str
        Short key from :pydata:`config.EMBEDDING_MODELS`.

    Raises
    ------
    ValueError
        If *model_key* is not recognised.
    """
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Choose from {list(EMBEDDING_MODELS)}"
        )
    name = EMBEDDING_MODELS[model_key]
    print(f"[embed] Loading {name}")
    return SentenceTransformer(name)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode *texts* into L2-normalised dense vectors.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), dim)`` float32 embeddings.
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

# FAISS index (dense vector search)

def build_faiss_index(
    embeddings: np.ndarray,
    use_hnsw: bool = True,
) -> faiss.Index:
    """Build a FAISS index over pre-computed embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        ``(N, D)`` L2-normalised vectors.
    use_hnsw : bool
        ``True`` → HNSW graph (fast ANN); ``False`` → flat exact search.
    """
    dim = embeddings.shape[1]
    if use_hnsw:
        index = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M)
        index.hnsw.efConstruction = FAISS_EF_CONSTRUCT
        index.hnsw.efSearch = FAISS_EF_SEARCH
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


# BM25 index (sparse lexical search)

# TODO: Change for a better tokenizer
def tokenize(text: str) -> list[str]:
    """Whitespace + lower-case tokenisation for BM25."""
    return text.lower().split()


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """Build a BM25Okapi sparse-lexical index."""
    return BM25Okapi([tokenize(t) for t in texts])


# Persistence

def save_indices(
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    model_key: str,
) -> None:
    """Persist FAISS index, BM25 index, and chunk list to ``INDEX_DIR``."""
    prefix = INDEX_DIR / model_key
    faiss.write_index(faiss_index, str(prefix) + "_faiss.index")
    with open(str(prefix) + "_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(str(prefix) + "_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"[index] Saved to {INDEX_DIR}/")


def load_indices(
    model_key: str,
) -> tuple[faiss.Index, BM25Okapi, list[Chunk]]:
    """Load previously-saved indices from ``INDEX_DIR``.

    Handles pickle files that stored ``Chunk`` under ``__main__`` or
    under the old ``embedding_indexing`` module namespace.
    """
    prefix = INDEX_DIR / model_key
    faiss_index = faiss.read_index(str(prefix) + "_faiss.index")

    with open(str(prefix) + "_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(str(prefix) + "_chunks.pkl", "rb") as f:
        chunks = _compat_load(f)

    return faiss_index, bm25, chunks

# Is this class really necessary?
class _ChunkUnpickler(pickle.Unpickler):
    """Redirect old Chunk class references to ``data_models.Chunk``."""

    def find_class(self, module: str, name: str):
        if name == "Chunk":
            return Chunk
        return super().find_class(module, name)


def _compat_load(f):
    """Load a pickle with backward-compatible Chunk resolution."""
    return _ChunkUnpickler(f).load()


# End-to-end build pipeline

def build_index(csv_path: Path, model_key: str = DEFAULT_MODEL_KEY) -> None:
    """Load CSV → embed → build indices → save.

    Parameters
    ----------
    csv_path : Path
        Evidence CSV produced by ``chunking_evidence.py``.
    model_key : str
        Embedding model short key.
    """
    print(f"[build] Loading chunks from {csv_path}")
    chunks = load_chunks(csv_path)
    texts = [c.text for c in chunks]
    print(f"[build] {len(chunks)} chunks loaded")

    model = get_embed_model(model_key)
    print("[build] Generating embeddings …")
    embeddings = embed_texts(texts, model)
    print(f"[build] Embeddings: {embeddings.shape}")

    print("[build] Building FAISS HNSW index")
    fi = build_faiss_index(embeddings)

    print("[build] Building BM25 index")
    bm = build_bm25_index(texts)

    save_indices(fi, bm, chunks, model_key)
    print("[build] Done")


def build_merged_index(
    *csv_paths: Path,
    model_key: str = DEFAULT_MODEL_KEY,
) -> None:
    """Merge multiple evidence CSVs, then build and save indices.

    Parameters
    ----------
    *csv_paths : Path
        One or more evidence CSV files.
    model_key : str
        Embedding model short key.
    """
    print(f"[build] Merging {len(csv_paths)} CSV files")
    chunks = load_and_merge_chunks(*csv_paths)
    texts = [c.text for c in chunks]
    print(f"[build] {len(chunks)} unique chunks after merge")

    model = get_embed_model(model_key)
    print("[build] Generating embeddings …")
    embeddings = embed_texts(texts, model)
    print(f"[build] Embeddings: {embeddings.shape}")

    print("[build] Building FAISS HNSW index")
    fi = build_faiss_index(embeddings)

    print("[build] Building BM25 index")
    bm = build_bm25_index(texts)

    save_indices(fi, bm, chunks, model_key)
    print("[build] Done")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build embedding + BM25 indices from evidence CSVs"
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path, nargs="+",
        help="One or more evidence CSV files",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_KEY,
        choices=list(EMBEDDING_MODELS),
    )
    args = parser.parse_args()

    if len(args.input) == 1:
        build_index(args.input[0], args.model)
    else:
        build_merged_index(*args.input, model_key=args.model)
