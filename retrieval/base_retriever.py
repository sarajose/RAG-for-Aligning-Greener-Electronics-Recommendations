"""
Abstract base class for all retriever implementations.

Every retriever exposes the same ``retrieve()`` interface so that
evaluation code can swap strategies without changes.
"""

from abc import ABC, abstractmethod
from data_models import RetrievalResult

class BaseRetriever(ABC):
    """Common interface for BM25, Dense, Hybrid, and Reranked retrievers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label used in reports and charts."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Return the top-*k* chunks for *query*.

        Parameters
        ----------
        query : str
            Free-text recommendation or question.
        top_k : int
            Number of results to return.

        Returns
        -------
        RetrievalResult
        """
  
