"""
RAG classification and LLM-as-judge package.

Modules
-------
- ``prompts``    — all prompt templates (classifier + judge)
- ``classifier`` — open-source LLM alignment classifier
- ``llm_judge``  — second-LLM evaluation of classification quality
"""

from rag.classifier import AlignmentClassifier
from rag.llm_judge import LLMJudge

__all__ = ["AlignmentClassifier", "LLMJudge"]
