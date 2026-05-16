"""
Domain data models shared across the RAG policy-alignment pipeline.
All objects are plain dataclasses with no framework dependencies so
they serialise easily to CSV and can be pickled for index storage
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

# Core domain objects
@dataclass
class Chunk:
    """A single legal-provision chunk from the evidence corpus.
    Attributes:
    id : str
        Stable hash-based identifier (document|article|para|hash).
    document : str
        Short document name
    source_file : str
        Original HTML filename
    version : str
        Date or version string extracted from the filename.
    chapter : str
        Chapter / title heading the chunk belongs to.
    article : str
        Article number.
    article_subtitle : str
        Article sub-heading, if any.
    paragraph : str
        Paragraph number within the article.
    char_offset : int
        Character offset in the source file.
    text : str
        Full provision text (the retrieval unit).
    article_text : str
        Full concatenated text of every paragraph in the parent article.
        Used as the generation context (parent-child chunking pattern).
        Empty string for chunks loaded from older CSVs without this column.
    """

    id: str
    document: str
    source_file: str
    version: str
    chapter: str
    article: str
    article_subtitle: str
    paragraph: str
    char_offset: int
    text: str
    article_text: str = field(default="")

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return asdict(self)

@dataclass
class Recommendation:
    """An sustainability recommendation to evaluate
    Attributes:
    section (str): Top-level section heading from the source document
    subsection (str) Sub-section heading
    title (str) Short title or parent context.
    text (str) The recommendation body text used as a retrieval query.
    """
    section: str
    subsection: str
    title: str
    text: str

@dataclass
class GoldStandardEntry:
    """One row from the manually-annotated gold standard.
    Each entry links a recommendation/statement
    to a relevant EU document at the document level
    """
    paper: str
    source_page: str
    source_line: str
    recommendation_text: str
    source_snippet_original: str
    recommendation_or_statement: str        
    doc_short_name: str                     
    doc_type: str
    doc_ref_num: str
    doc_reference_raw_excerpt: str
    evidence_span: str
    reference_basis: str                    
    needs_review: str
    context_excerpt: str
    alignment_label: Optional[str] = None   # future: ALIGNMENT_LABELS

# Pipeline result objects
@dataclass
class RetrievalResult:
    """Top-k retrieval output for a single query.
    Attributes:
    query (str): The input recommendation / question.
    ranked_chunks (list[Chunk]): Retrieved chunks ordered by relevance (best first).
    scores (list[float]): Corresponding relevance scores (same order).
    """
    query: str
    ranked_chunks: list[Chunk]
    scores: list[float]
    evidence_groups: list[str] = field(default_factory=list)
    retrieval_mode: str = "flat_baseline"

@dataclass
class ClassificationResult:
    """LLM alignment-classification output for one recommendation.
    Attributes
    recommendation (str): Input recommendation text
    label (str): Predicted alignment label
    justification (str): Evidence-based reasoning produced by the LLM
    cited_chunk_ids (list[str]): Chunk IDs the LLM cited as evidence.
    retrieved_chunks (list[Chunk]): Evidence window sent to the LLM.
    raw_llm_response (str): Raw LLM output for debugging / auditing.
    """
    recommendation: str
    label: str
    justification: str
    cited_chunk_ids: list[str]
    retrieved_chunks: list[Chunk]
    raw_llm_response: str = ""

# Metric containers
@dataclass
class RetrievalMetrics:
    """Aggregated retrieval-quality metrics at a given cut-off k
    All scores are averaged over the query set unless stated otherwise.
    When each query has exactly one relevant document (as in our gold
    standard). Hit@k and MRR.
    """
    k: int
    hit_rate: float             # fraction of queries with ≥ 1 hit in top-k
    recall: float               # mean fraction of relevant docs found
    precision: float            # mean (relevant in top-k) / k
    mrr: float                  # Mean Reciprocal Rank
    map_score: float            # Mean Average Precision
    ndcg: float                 # Normalised Discounted Cumulative Gain
    num_queries: int
    mean_rank: float = float("inf")  # mean rank of first relevant item
    chunk_hit_rate: float = 0.0      # fraction of queries where top-1 raw chunk is from correct doc

@dataclass
class ClassificationMetrics:
    """Aggregated alignment-label quality metrics."""
    accuracy: float
    macro_f1: float
    weighted_f1: float
    cohens_kappa: float
    per_class: dict[str, dict[str, float]]  # label
    confusion_matrix: list[list[int]]
    labels: list[str]
    num_samples: int