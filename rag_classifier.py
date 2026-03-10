"""
Backward-compatible wrapper for the alignment classifier.

The actual implementation now lives in :pymod:`rag.classifier` (open-source
LLM via HuggingFace transformers).  This file re-exports the class and
helpers so that existing imports continue to work::

    from rag_classifier import AlignmentClassifier   # still works
    from rag.classifier import AlignmentClassifier    # new canonical path
"""

# Re-export everything from the new location
from rag.classifier import AlignmentClassifier, _parse_json_response as parse_llm_response
from rag.prompts import format_evidence_block, build_classifier_messages

__all__ = ["AlignmentClassifier", "parse_llm_response", "format_evidence_block"]
