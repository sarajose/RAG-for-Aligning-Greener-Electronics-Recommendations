"""
LLM-as-judge evaluation of alignment classifications.

Uses a **different** open-source LLM from the classifier to avoid
self-evaluation bias (Zheng et al., "Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena", NeurIPS 2023).

Default judge model: **mistralai/Mistral-7B-Instruct-v0.3** —
a different model family from the Qwen-based classifier.

Each classification receives three sub-scores (1-5) and an overall
score:

- **Label correctness** — is the alignment label appropriate?
- **Justification quality** — is the reasoning grounded in evidence?
- **Evidence usage** — are the cited provisions relevant and sufficient?
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_models import Chunk, ClassificationResult
from rag.prompts import build_judge_messages

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass
class JudgeResult:
    """Output of the LLM-as-judge evaluation for one classification."""

    recommendation: str
    predicted_label: str
    label_score: int          # 1-5
    justification_score: int  # 1-5
    evidence_score: int       # 1-5
    overall_score: float      # average
    reasoning: str
    raw_response: str = ""


def _parse_judge_response(raw: str) -> dict:
    """Parse judge JSON output, with fallback."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return {
            "label_score": int(data.get("label_score", 1)),
            "justification_score": int(data.get("justification_score", 1)),
            "evidence_score": int(data.get("evidence_score", 1)),
            "overall_score": float(data.get("overall_score", 1.0)),
            "reasoning": data.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse judge JSON — assigning score 1")
        return {
            "label_score": 1,
            "justification_score": 1,
            "evidence_score": 1,
            "overall_score": 1.0,
            "reasoning": f"PARSE_ERROR: {raw[:200]}",
        }


class LLMJudge:
    """Second-LLM evaluator for alignment classification quality.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the judge.
    quantize_4bit : bool
        Use 4-bit quantisation (requires ``bitsandbytes``).
    max_new_tokens : int
        Maximum tokens to generate.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_JUDGE_MODEL,
        quantize_4bit: bool = False,
        device_map: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        logger.info("Loading judge model: %s", model_name)
        print(f"[judge] Loading {model_name} …")

        load_kwargs: dict = {
            "device_map": device_map,
            "torch_dtype": "auto",
        }

        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading without quantisation"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs,
        )
        print("[judge] Model loaded")

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run inference and return raw generated text."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text], return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def evaluate(
        self,
        classification: ClassificationResult,
    ) -> JudgeResult:
        """Evaluate a single classification with the judge LLM.

        Parameters
        ----------
        classification : ClassificationResult
            Output from the alignment classifier.

        Returns
        -------
        JudgeResult
        """
        messages = build_judge_messages(
            recommendation=classification.recommendation,
            chunks=classification.retrieved_chunks,
            label=classification.label,
            justification=classification.justification,
            cited_chunk_ids=classification.cited_chunk_ids,
        )
        raw = self._generate(messages)
        parsed = _parse_judge_response(raw)

        return JudgeResult(
            recommendation=classification.recommendation,
            predicted_label=classification.label,
            label_score=parsed["label_score"],
            justification_score=parsed["justification_score"],
            evidence_score=parsed["evidence_score"],
            overall_score=parsed["overall_score"],
            reasoning=parsed["reasoning"],
            raw_response=raw,
        )

    def evaluate_batch(
        self,
        classifications: list[ClassificationResult],
    ) -> list[JudgeResult]:
        """Evaluate a batch of classifications.

        Parameters
        ----------
        classifications : list[ClassificationResult]

        Returns
        -------
        list[JudgeResult]
        """
        results: list[JudgeResult] = []
        for i, cls_result in enumerate(classifications, 1):
            print(f"  [judge {i}/{len(classifications)}] "
                  f"{cls_result.recommendation[:60]}…")
            results.append(self.evaluate(cls_result))
        return results
