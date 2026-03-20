"""
Open-source LLM alignment classifier.

Default model: **Qwen/Qwen2.5-7B-Instruct** — top-ranked open-source
model for instruction-following and structured JSON output on the MTEB
leaderboard (as of early 2025).  Supports 4-bit quantisation via
``bitsandbytes`` to run on GPUs with ≥8 GB VRAM.

Usage::

    from rag.classifier import AlignmentClassifier

    clf = AlignmentClassifier()                     # default: Qwen2.5-7B
    clf = AlignmentClassifier(quantize_4bit=True)   # 4-bit quantised
    result = clf.classify(recommendation_text, retrieved_chunks)
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    ALIGNMENT_LABELS,
    LLM_CPU_MAX_MEMORY,
    LLM_GPU_MAX_MEMORY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_OFFLOAD_DIR,
    LLM_QUANTIZE_4BIT,
    LLM_TEMPERATURE,
)
from data_models import Chunk, ClassificationResult
from rag.prompts import build_classifier_messages

logger = logging.getLogger(__name__)

# Default open-source model
DEFAULT_CLASSIFIER_MODEL = LLM_MODEL


def _parse_json_response(raw: str) -> dict:
    """Parse strict JSON from LLM output, tolerating markdown fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON — returning raw text.")
        return {
            "label": "PARSE_ERROR",
            "justification": raw,
            "cited_chunk_ids": [],
        }


def _best_matching_label(candidate: str) -> str:
    """Fuzzy-match a possibly malformed label to the closest valid one."""
    low = candidate.lower().strip()
    for label in ALIGNMENT_LABELS:
        if label.lower() in low or low in label.lower():
            return label
    return candidate


class AlignmentClassifier:
    """Open-source LLM alignment classifier.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    quantize_4bit : bool
        Use 4-bit quantisation (requires ``bitsandbytes``).
    device_map : str
        Device placement strategy (``"auto"`` uses GPU if available).
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature (0.0 = greedy / deterministic).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLASSIFIER_MODEL,
        quantize_4bit: bool = LLM_QUANTIZE_4BIT,
        device_map: str = "auto",
        max_new_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        offload_folder: Path = LLM_OFFLOAD_DIR / "classifier",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info("Loading classifier model: %s", model_name)
        print(f"[classifier] Loading {model_name} …")

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
                print("[classifier] 4-bit quantisation enabled")
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
                "4-bit model load failed (%s) — retrying without quantisation", exc,
            )
            print("[classifier] 4-bit load failed; retrying without quantisation …")
            load_kwargs.pop("quantization_config", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs,
            )
        print("[classifier] Model loaded")

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Run inference and return the raw generated text."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text], return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False
            # Use canonical greedy defaults to avoid generation-config warnings.
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 50

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def classify(
        self,
        recommendation: str,
        chunks: list[Chunk],
    ) -> ClassificationResult:
        """Classify alignment of *recommendation* against *chunks*.

        Parameters
        ----------
        recommendation : str
            Sustainability recommendation text.
        chunks : list[Chunk]
            Pre-retrieved evidence chunks (typically 5–10).

        Returns
        -------
        ClassificationResult
        """
        messages = build_classifier_messages(recommendation, chunks)
        raw = self._generate(messages)
        parsed = _parse_json_response(raw)

        label = parsed.get("label", "PARSE_ERROR")
        if label not in ALIGNMENT_LABELS:
            label = _best_matching_label(label)

        return ClassificationResult(
            recommendation=recommendation,
            label=label,
            justification=parsed.get("justification", ""),
            cited_chunk_ids=parsed.get("cited_chunk_ids", []),
            retrieved_chunks=chunks,
            raw_llm_response=raw,
        )
