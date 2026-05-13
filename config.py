"""
Configuration for the RAG policy-alignment pipeline.
Centralises paths, model identifiers, alignment labels 
and hyperparameters.
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
OUTPUT_DIR = BASE_DIR / "outputs" 
INDEX_DIR = OUTPUT_DIR / "indices"
GOLD_STANDARD_DIR = DATA_DIR / "gold_standard_doc_level"
BENCHMARK_DIR = BASE_DIR / "benchmarks"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
DOCS_DIR = BASE_DIR / "docs"

for _d in (OUTPUT_DIR, INDEX_DIR, GOLD_STANDARD_DIR, BENCHMARK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Cache/download settings
HF_CACHE_DIR = OUTPUT_DIR / "hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Reduce CUDA memory fragmentation (to run in small GPUs)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    # expandable_segments lets PyTorch release memory back to the OS between stages
    "expandable_segments:True,max_split_size_mb:128",
)

# Default file paths
EVIDENCE_CSV = OUTPUT_DIR / "evidence.csv"
EVIDENCE_REC_CSV = OUTPUT_DIR / "evidence_recommendation.csv"
GOLD_STANDARD_CSV = GOLD_STANDARD_DIR / "gold_standard.csv"
WHITEPAPER_RECOMMENDATIONS_CSV = DATA_DIR / "recommendations_whitepaper" / "recommendations_v2.csv"

# Embedding models
EMBEDDING_MODELS: dict[str, str] = {
    "bge-m3":     "BAAI/bge-m3",
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",
}

# Cross-encoder reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_KEY = "bge-m3"

# SPLADE sparse retriever
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
SPLADE_MAX_LENGTH = 256

# Retrieval modes exposed in CLI
RETRIEVAL_MODES: list[str] = [
    "flat_baseline",
    "split_evidence_retrieval",
]
DEFAULT_RETRIEVAL_MODE = "flat_baseline"

# LLM_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"   # lightweight classifier
# JUDGE_MODEL = "HuggingFaceTB/SmolLM3-3B"  # lightweight judge

LLM_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3" # (needs >8 GB)
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"          # (needs >8 GiB)
LLM_TEMPERATURE = 0.0          # deterministic for reproducibility
LLM_MAX_TOKENS = 4096          # for paragraph justifications
LLM_MAX_INPUT_TOKENS = 16384   # long context (allows full article_text)
LLM_QUANTIZE_4BIT = True         # saves memory with negligible quality loss
JUDGE_QUANTIZE_4BIT = True       

# Judge generation limits.
JUDGE_MAX_NEW_TOKENS = 2048    # full per-criterion reasoning
JUDGE_MAX_INPUT_TOKENS = 16384 # evidence + classification + system prompt fits

LLM_GPU_MAX_MEMORY = "40GiB"   
LLM_CPU_MAX_MEMORY = "64GiB"

# Maximum characters per evidence chunk fed to the LLM
# EVIDENCE_MAX_CHARS_PER_CHUNK: int | None = None # (uses full article_text)
EVIDENCE_MAX_CHARS_PER_CHUNK: int | None = 6000  # fallback if memory issues
LLM_OFFLOAD_DIR = OUTPUT_DIR / "offload"
LLM_OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Alignment labels
ALIGNMENT_LABELS: list[str] = [
    "Aligned",
    "Conditional",
    "Conflicting",
    "No explicit legal basis",
    "Beyond compliance",
]

# Retrieval hyper-parameters
DEFAULT_TOP_K = 10             # hybrid candidates before reranking
DEFAULT_RERANK_TOP = 7         # results kept after cross-encoder
FAISS_HNSW_M = 32              # bi-directional links per node
FAISS_EF_CONSTRUCT = 40        # construction search depth
FAISS_EF_SEARCH = 16           # query-time search depth
RRF_K = 60                     # RRF smoothing constant

# Evaluation K values
EVAL_K_VALUES: list[int] = [1, 3, 5, 10, 20]

# Document name normalisation
# Maps a canonical short name to patterns that
# may appear in either the gold-standard doc_short_name 
# or the evidence document column. The first matching pattern wins.

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
    Parameters: raw : str
        A document name from either the gold standard or evidence CSV.
    Returns: str short name (e.g. "ESPR", "WEEE")
    """
    low = raw.lower().strip()
    # Exact match shortcuts for abbreviated corpus names
    _EXACT: dict[str, str] = {"net": "Net-Zero", "wee": "WEEE"}
    if low in _EXACT:
        return _EXACT[low]
    for canonical, patterns in DOC_CANONICAL_MAP.items():
        for pat in patterns:
            if pat in low:
                return canonical
    return raw.strip().title()


_BINDING_LAW_DOCS: set[str] = {
    "ESPR",
    "Ecodesign Directive",
    "REACH",
    "RoHS",
    "WEEE",
    "Battery Regulation",
    "CSRD",
    "CSDDD",
    "CRMA",
    "Green Claims",
    "Waste Framework",
    "Right to Repair",
    "Net-Zero",
    "PPWR",
    "CBAM",
    "EU Chips Act",
    "Taxonomy",
    "Conflict Minerals",
}


def evidence_group_for_document(document_name: str) -> str:
    """Map a document to one of two evidence groups used by split retrieval."""
    canonical = normalise_doc_name(document_name)
    if canonical in _BINDING_LAW_DOCS:
        return "binding_law"
    return "policy_or_recommendation_docs"