from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import BinaryIO

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

from config import (
    DEFAULT_MODEL_KEY,
    FAISS_EF_CONSTRUCT,
    FAISS_EF_SEARCH,
    FAISS_HNSW_M,
    INDEX_DIR,
)
from data_models import Chunk
from indexing.chunks import load_and_merge_chunks, load_chunks
from indexing.embeddings import embed_texts, get_embed_model


def build_faiss_index(embeddings: np.ndarray, use_hnsw: bool = True) -> faiss.Index:
    """Build FAISS index over normalized embedding vectors."""
    dim = embeddings.shape[1]
    if use_hnsw:
        index = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M)
        index.hnsw.efConstruction = FAISS_EF_CONSTRUCT
        index.hnsw.efSearch = FAISS_EF_SEARCH
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings.astype(np.float32))
    return index


# ── BM25 tokenizer with NLTK stopwords + Porter stemming ────────────────────
# Lazy-initialised so the import cost is only paid on first use.
_BM25_STOPWORDS: frozenset | None = None
_BM25_STEMMER = None


def _init_bm25_tools() -> tuple:
    """Lazy-load NLTK stopwords and Porter stemmer (once per process)."""
    global _BM25_STOPWORDS, _BM25_STEMMER
    if _BM25_STOPWORDS is not None:
        return _BM25_STOPWORDS, _BM25_STEMMER
    try:
        import nltk
        from nltk.corpus import stopwords as sw
        from nltk.stem import PorterStemmer

        try:
            words = sw.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = sw.words("english")

        _BM25_STOPWORDS = frozenset(words)
        _BM25_STEMMER = PorterStemmer()
        logger.info("BM25 tokenizer: NLTK stopwords (%d) + Porter stemmer loaded",
                    len(_BM25_STOPWORDS))
    except ImportError:
        logger.warning(
            "nltk not installed — BM25 will use whitespace-only tokenisation. "
            "Install with: pip install nltk"
        )
        _BM25_STOPWORDS = frozenset()
        _BM25_STEMMER = None
    return _BM25_STOPWORDS, _BM25_STEMMER


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 with stopword removal and Porter stemming.

    Stopword removal reduces noise from high-frequency function words
    (the, of, in, …) that appear in every legal provision.  Porter stemming
    groups morphological variants (recycling/recycled/recyclable) so BM25
    can match across inflected forms.  Legal normative terms (shall, may,
    must) are intentionally kept — they are not in NLTK's English stopwords.
    """
    stopwords, stemmer = _init_bm25_tools()
    tokens = text.lower().split()
    result: list[str] = []
    for tok in tokens:
        tok = tok.strip(".,;:\"'()[]")  # strip trailing punctuation
        if len(tok) < 2:
            continue
        if tok in stopwords:
            continue
        if stemmer is not None:
            tok = stemmer.stem(tok)
        if tok:
            result.append(tok)
    return result


def build_bm25_index(texts: list[str]) -> BM25Okapi:
    """Build BM25 index from raw texts."""
    return BM25Okapi([tokenize(text) for text in texts])


def save_indices(
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    model_key: str,
) -> None:
    """Persist FAISS, BM25, and chunk metadata to disk."""
    prefix = INDEX_DIR / model_key
    faiss.write_index(faiss_index, str(prefix) + "_faiss.index")
    with open(str(prefix) + "_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(str(prefix) + "_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"[index] Saved to {INDEX_DIR}/")


class _ChunkUnpickler(pickle.Unpickler):
    """Redirect legacy Chunk class references to data_models.Chunk."""

    def find_class(self, module: str, name: str):
        if name == "Chunk":
            return Chunk
        return super().find_class(module, name)


def _compat_load(handle: BinaryIO) -> list[Chunk]:
    """Load pickle with backward-compatible Chunk resolution."""
    return _ChunkUnpickler(handle).load()


def load_indices(model_key: str) -> tuple[faiss.Index, BM25Okapi, list[Chunk]]:
    """Load persisted FAISS, BM25, and chunk metadata for a model key."""
    prefix = INDEX_DIR / model_key
    faiss_index = faiss.read_index(str(prefix) + "_faiss.index")

    with open(str(prefix) + "_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(str(prefix) + "_chunks.pkl", "rb") as f:
        chunks = _compat_load(f)

    return faiss_index, bm25, chunks


def build_index(csv_path: Path, model_key: str = DEFAULT_MODEL_KEY) -> None:
    """Load CSV, build embeddings and indices, then persist them."""
    print(f"[build] Loading chunks from {csv_path}")
    chunks = load_chunks(csv_path)
    texts = [chunk.text for chunk in chunks]
    print(f"[build] {len(chunks)} chunks loaded")

    model = get_embed_model(model_key)
    print("[build] Generating embeddings")
    embeddings = embed_texts(texts, model)
    print(f"[build] Embeddings: {embeddings.shape}")

    print("[build] Building FAISS HNSW index")
    faiss_index = build_faiss_index(embeddings)

    print("[build] Building BM25 index")
    bm25 = build_bm25_index(texts)

    save_indices(faiss_index, bm25, chunks, model_key)
    print("[build] Done")


def build_merged_index(*csv_paths: Path, model_key: str = DEFAULT_MODEL_KEY) -> None:
    """Merge multiple evidence CSVs, then build and persist indices."""
    print(f"[build] Merging {len(csv_paths)} CSV files")
    chunks = load_and_merge_chunks(*csv_paths)
    texts = [chunk.text for chunk in chunks]
    print(f"[build] {len(chunks)} unique chunks after merge")

    model = get_embed_model(model_key)
    print("[build] Generating embeddings")
    embeddings = embed_texts(texts, model)
    print(f"[build] Embeddings: {embeddings.shape}")

    print("[build] Building FAISS HNSW index")
    faiss_index = build_faiss_index(embeddings)

    print("[build] Building BM25 index")
    bm25 = build_bm25_index(texts)

    save_indices(faiss_index, bm25, chunks, model_key)
    print("[build] Done")
