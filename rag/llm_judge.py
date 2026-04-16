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
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    JUDGE_MAX_NEW_TOKENS,
    JUDGE_MODEL,
    JUDGE_QUANTIZE_4BIT,
    LLM_CPU_MAX_MEMORY,
    LLM_GPU_MAX_MEMORY,
    LLM_OFFLOAD_DIR,
)
from data_models import Chunk, ClassificationResult
from rag.prompts import build_judge_messages, build_judge_retry_messages

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = JUDGE_MODEL

# Reasoning models (e.g. SmolLM3) emit a <think>…</think> block before JSON.
# That block alone is typically 300-800 tokens; the JSON output adds ~150.
# 1024 ensures the model can finish both the thinking block and the JSON.
_JUDGE_MIN_NEW_TOKENS = 1024


@dataclass
class JudgeResult:
    """Output of the LLM-as-judge evaluation for one classification."""

    recommendation: str
    predicted_label: str
    label_score: int          # 1-5
    justification_score: int  # 1-5
    evidence_score: int       # 1-5
    completeness_score: int   # 1-5 (gap analysis + improvement suggestions)
    overall_score: float      # average of all four scores
    reasoning: str
    raw_response: str = ""


def _parse_judge_response(raw: str) -> dict:
    """Parse judge JSON output.

    Tries four strategies in order:
    1. Strict ``json.loads`` on the cleaned text.
    2. Parse an embedded JSON object inside wrapper text.
    3. Regex extraction of score fields from partial/truncated JSON.
    4. Hard fallback (all scores = 1) when all above fail.
    """
    text = raw.strip()
    # Reasoning models (e.g. SmolLM3) wrap chain-of-thought in <think>…</think>
    # before the actual JSON.  Two cases:
    # 1. Complete block: <think>…</think>JSON  →  strip the block, keep JSON.
    # 2. Truncated mid-think (no </think>): <think>…  →  nothing useful remains.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Strategy 1: strict JSON
    try:
        data = json.loads(text)
        label_s = int(data.get("label_score", 1))
        just_s = int(data.get("justification_score", 1))
        evid_s = int(data.get("evidence_score", 1))
        comp_s = int(data.get("completeness_score", 1))
        return {
            "label_score": label_s,
            "justification_score": just_s,
            "evidence_score": evid_s,
            "completeness_score": comp_s,
            "overall_score": round((label_s + just_s + evid_s + comp_s) / 4, 2),
            "reasoning": str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: recover a JSON object embedded in extra text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            label_s = int(data.get("label_score", 1))
            just_s = int(data.get("justification_score", 1))
            evid_s = int(data.get("evidence_score", 1))
            comp_s = int(data.get("completeness_score", 1))
            logger.warning("Judge JSON malformed — recovered embedded object.")
            return {
                "label_score": label_s,
                "justification_score": just_s,
                "evidence_score": evid_s,
                "completeness_score": comp_s,
                "overall_score": round((label_s + just_s + evid_s + comp_s) / 4, 2),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: regex extraction from partial/truncated JSON
    scores: dict[str, int] = {}
    for field in ("label_score", "justification_score", "evidence_score", "completeness_score"):
        m = re.search(rf'"{field}"\s*:\s*([1-5])', text)
        if m:
            scores[field] = int(m.group(1))
    reasoning_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    reasoning = reasoning_m.group(1) if reasoning_m else ""

    if scores:
        label_s = scores.get("label_score", 1)
        just_s = scores.get("justification_score", 1)
        evid_s = scores.get("evidence_score", 1)
        comp_s = scores.get("completeness_score", 1)
        logger.warning("Judge JSON malformed — recovered partial scores via regex: %s", scores)
        return {
            "label_score": label_s,
            "justification_score": just_s,
            "evidence_score": evid_s,
            "completeness_score": comp_s,
            "overall_score": round((label_s + just_s + evid_s + comp_s) / 4, 2),
            "reasoning": reasoning or "Scores extracted from malformed JSON response.",
        }

    # Strategy 4: hard fallback
    logger.warning("Failed to parse judge JSON — assigning score 1. Raw: %.120s", text)
    return {
        "label_score": 1,
        "justification_score": 1,
        "evidence_score": 1,
        "completeness_score": 1,
        "overall_score": 1.0,
        "reasoning": "PARSE_ERROR: Judge response was not valid JSON. Default scores were assigned.",
    }


def _contains_cjk(text: str) -> bool:
    """Return True when text contains CJK characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _normalize_reasoning_to_english(reasoning: str) -> str:
    """Guarantee English fallback reasoning for downstream CSV outputs."""
    if _contains_cjk(reasoning):
        return (
            "Reasoning contained non-English text; the judge result was retained, "
            "but the explanation was normalized to English."
        )
    return reasoning


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
        quantize_4bit: bool = JUDGE_QUANTIZE_4BIT,
        device_map: str = "auto",
        max_new_tokens: int = JUDGE_MAX_NEW_TOKENS,
        max_input_tokens: int = 4096,
        offload_folder: Path = LLM_OFFLOAD_DIR / "judge",
    ) -> None:
        self.model_name = model_name
        if max_new_tokens < _JUDGE_MIN_NEW_TOKENS:
            print(
                f"[judge] max_new_tokens={max_new_tokens} is too low to complete "
                f"the JSON structure; raising to {_JUDGE_MIN_NEW_TOKENS}."
            )
            max_new_tokens = _JUDGE_MIN_NEW_TOKENS
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

        logger.info("Loading judge model: %s", model_name)
        print(f"[judge] Loading {model_name} …")

        offload_folder = Path(offload_folder)
        offload_folder.mkdir(parents=True, exist_ok=True)

        max_memory: dict = {"cpu": LLM_CPU_MAX_MEMORY}
        if torch.cuda.is_available():
            max_memory[0] = LLM_GPU_MAX_MEMORY

        load_kwargs: dict = {
            "device_map": device_map,
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "offload_state_dict": True,
            "offload_folder": str(offload_folder),
            "offload_buffers": True,
            "max_memory": max_memory,
        }

        quantization_enabled = False
        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                quantization_enabled = True
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading without quantisation"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs,
            )
        except Exception as exc:
            if not quantization_enabled:
                raise
            logger.warning(
                "4-bit judge load failed (%s) — retrying without quantisation", exc,
            )
            print("[judge] 4-bit load failed; retrying without quantisation …")
            load_kwargs.pop("quantization_config", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs,
            )

        # Keep generation deterministic and avoid sampling-flag warnings.
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
        print("[judge] Model loaded")

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run inference and return raw generated text."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,   # greedy — deterministic and reproducible
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
        # Keep judge prompts compact to avoid GPU memory spikes.
        judge_chunks = [Chunk(**{**c.to_dict(), "article_text": ""}) for c in classification.retrieved_chunks]
        messages = build_judge_messages(
            recommendation=classification.recommendation,
            chunks=judge_chunks,
            label=classification.label,
            justification=classification.justification[:2000],
            cited_chunk_ids=classification.cited_chunk_ids,
        )
        raw = self._generate(messages)
        parsed = _parse_judge_response(raw)

        # Retry once with a stripped-down prompt if primary parse failed completely
        if parsed["reasoning"].startswith("PARSE_ERROR"):
            logger.info("Judge parse failed — retrying with simplified prompt")
            retry_messages = build_judge_retry_messages(
                label=classification.label,
                justification=classification.justification,
            )
            raw = self._generate(retry_messages)
            parsed = _parse_judge_response(raw)

        return JudgeResult(
            recommendation=classification.recommendation,
            predicted_label=classification.label,
            label_score=parsed["label_score"],
            justification_score=parsed["justification_score"],
            evidence_score=parsed["evidence_score"],
            completeness_score=parsed["completeness_score"],
            overall_score=parsed["overall_score"],
            reasoning=_normalize_reasoning_to_english(parsed["reasoning"]),
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results
