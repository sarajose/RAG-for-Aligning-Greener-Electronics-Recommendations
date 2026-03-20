"""
Configuration for the RAG policy-alignment pipeline.

Centralises paths, model identifiers, alignment labels, and tunable
hyper-parameters.  Every other module imports from here so that
nothing is hard-coded elsewhere.
"""

import os
from pathlib import Path

# Project paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
OUTPUT_DIR = Path("D:/outputs")  # Use D: drive for outputs (C: version below is commented)
# OUTPUT_DIR = BASE_DIR / "outputs"  # Use C: drive for outputs
INDEX_DIR = OUTPUT_DIR / "indices"
GOLD_STANDARD_DIR = DATA_DIR / "gold_standard_doc_level"
BENCHMARK_DIR = BASE_DIR / "benchmarks"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
DOCS_DIR = BASE_DIR / "docs"

for _d in (OUTPUT_DIR, INDEX_DIR, GOLD_STANDARD_DIR, BENCHMARK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Hugging Face cache/download settings (Windows-friendly)
HF_CACHE_DIR = OUTPUT_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Default file paths

EVIDENCE_CSV = OUTPUT_DIR / "evidence.csv"  # C: drive
EVIDENCE_REC_CSV = OUTPUT_DIR / "evidence_recommendation.csv"  # C: drive
GOLD_STANDARD_CSV = GOLD_STANDARD_DIR / "gold_standard.csv"
WHITEPAPER_RECOMMENDATIONS_CSV = DATA_DIR / "recommendations_whitepaper" / "recommendations_v2.csv"
# WHITEPAPER_RECOMMENDATIONS_CSV = Path(r"")

# Embedding models

EMBEDDING_MODELS: dict[str, str] = {
    "bge-m3":     "BAAI/bge-m3",
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",
    "mpnet":      "sentence-transformers/all-mpnet-base-v2",
    "minilm":     "sentence-transformers/all-MiniLM-L6-v2",
}
DEFAULT_MODEL_KEY = "bge-m3"

# SPLADE sparse retriever
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
SPLADE_MAX_LENGTH = 256

# Cross-encoder reranker

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM (alignment classification) — open-source models

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"         # classifier
JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # independent LLM-as-judge
LLM_TEMPERATURE = 0.0          # deterministic for reproducibility
LLM_MAX_TOKENS = 1024
LLM_QUANTIZE_4BIT = True         # safer default for limited VRAM
JUDGE_QUANTIZE_4BIT = True       # judge is also 7B-scale
LLM_GPU_MAX_MEMORY = "8GiB"
LLM_CPU_MAX_MEMORY = "24GiB"
#LLM_OFFLOAD_DIR = OUTPUT_DIR / "offload"  # C: drive version
LLM_OFFLOAD_DIR = Path("D:/outputs/offload")  # D: drive for offload
LLM_OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Alignment labels (to be denfined, just as a placeholder)

ALIGNMENT_LABELS: list[str] = [
    "Aligned",
    "Conditional",
    "Conflicting",
    "No explicit legal basis",
    "Beyond compliance",
]

# Retrieval hyper-parameters

DEFAULT_TOP_K = 10             # hybrid candidates before reranking
DEFAULT_RERANK_TOP = 5         # results kept after cross-encoder
FAISS_HNSW_M = 32              # bi-directional links per node
FAISS_EF_CONSTRUCT = 40        # construction search depth
FAISS_EF_SEARCH = 16           # query-time search depth
RRF_K = 60                     # RRF smoothing constant (according to Reciprocal Rank Fusion 
                               # outperforms Condorcet and individual Rank Learning Methods paper)

# Evaluation K values
EVAL_K_VALUES: list[int] = [1, 3, 5, 10, 20]

# ── Document-name normalisation ─────────────────────────────────────────────
# Maps a canonical short name → patterns (case-insensitive sub-strings) that
# may appear in either the gold-standard ``doc_short_name`` or the evidence
# ``document`` column.  The first matching pattern wins, so more specific
# entries should precede generic ones.

DOC_CANONICAL_MAP: dict[str, list[str]] = {
    "ESPR":                 ["espr", "ecodesign for sustainable products"],
    "Ecodesign Directive":  ["ecodesign directive", "2009/125"],
    "REACH":                ["reach", "1907/2006"],
    "RoHS":                 ["rohs", "2011/65"],
    "WEEE":                 ["weee", "wee-", "wee ", "2012/19"],
    "Battery Regulation":   ["battery", "2023/1542", "eu battery"],
    "CSRD":                 ["csrd", "2022/2464"],
    "CSDDD":                ["csddd", "2024/1760"],
    "CEAP":                 ["circular economy action plan", "ceap",
                             "com(2020)98"],
    "SSbD":                 ["ssbd", "safe-and-sustainable-by-design",
                             "2022/2510"],
    "Green Deal":           ["european green deal", "green deal",
                             "com(2019)640"],
    "CRMA":                 ["critical raw materials", "crma", "2024/1252"],
    "Green Claims":         ["green claims"],
    "Waste Framework":      ["waste framework", "2008/98"],
    "Right to Repair":      ["right to repair", "2024/1799"],
    "Chemicals Strategy":   ["chemicals strategy for sustainability"],
    "Net-Zero":             ["net-zero industry act", "net-zero", "net zero"],
    "PPWR":                 ["ppwr", "packaging and packaging waste"],
    "CBAM":                 ["cbam", "carbon border"],
    "EU Chips Act":         ["chips act", "eu chips"],
    "Omnibus":              ["omnibus"],
    "Taxonomy":             ["taxonomy", "2020/852"],
    "Single Market":        ["single market strategy"],
    "Conflict Minerals":    ["conflict minerals", "2017/821"],
    "Competitive Compass":  ["competitive compass"],
    "Clean Industrial Deal": ["clean industrial deal"],
}


def normalise_doc_name(raw: str) -> str:
    """Return the canonical short name for any document reference string.

    Matching is case-insensitive.  Returns the original string (stripped
    and title-cased) when no canonical pattern matches.

    Parameters
    ----------
    raw : str
        A document name from either the gold standard or evidence CSV.

    Returns
    -------
    str
        Canonical short name (e.g. ``"ESPR"``, ``"WEEE"``).
    """
    low = raw.lower().strip()
    # Exact-match shortcuts for abbreviated corpus names
    _EXACT: dict[str, str] = {"net": "Net-Zero", "wee": "WEEE"}
    if low in _EXACT:
        return _EXACT[low]
    for canonical, patterns in DOC_CANONICAL_MAP.items():
        for pat in patterns:
            if pat in low:
                return canonical
    return raw.strip().title()