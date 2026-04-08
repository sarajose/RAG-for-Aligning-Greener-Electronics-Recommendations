from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_models import Chunk, ClassificationResult
from pipeline_io import save_judge_results_csv
from rag.llm_judge import LLMJudge


def _split_semicolon(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def _parse_chunk_header(text: str) -> tuple[str, str]:
    # Expected format like: [PPWR | Article 3]
    first_line = (text or "").splitlines()[0].strip()
    m = re.match(r"^\[(.*?)\s*\|\s*(.*?)\]", first_line)
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(2).strip()


def _build_chunks(row: pd.Series) -> list[Chunk]:
    texts_raw = str(row.get("top_chunk_texts", "") or "")
    chunk_texts = [t.strip() for t in texts_raw.split("\n---\n") if t.strip()]
    chunk_ids = _split_semicolon(row.get("top_chunk_ids", ""))

    chunks: list[Chunk] = []
    for i, chunk_text in enumerate(chunk_texts):
        chunk_id = chunk_ids[i] if i < len(chunk_ids) else f"generated_{i+1}"
        document, article = _parse_chunk_header(chunk_text)
        chunks.append(
            Chunk(
                id=chunk_id,
                document=document,
                source_file="",
                version="",
                chapter="",
                article=article,
                article_subtitle="",
                paragraph="",
                char_offset=0,
                text=chunk_text,
            )
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate judge CSV from existing classifications CSV")
    parser.add_argument("--input", type=Path, required=True, help="Classifications CSV (e.g., outputs/qwen_classifications.csv)")
    parser.add_argument("--output", type=Path, required=True, help="Judge CSV output path")
    parser.add_argument("--model", type=str, default=None, help="Optional judge model override")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation cap for judge output")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    classifications: list[ClassificationResult] = []
    for _, row in df.iterrows():
        recommendation = str(row.get("recommendation_query", "") or "")
        label = str(row.get("alignment_label", "") or "")
        justification = str(row.get("justification", "") or "")
        cited_chunk_ids = _split_semicolon(row.get("cited_chunk_ids", ""))
        retrieved_chunks = _build_chunks(row)

        if not recommendation.strip():
            continue

        classifications.append(
            ClassificationResult(
                recommendation=recommendation,
                label=label,
                justification=justification,
                cited_chunk_ids=cited_chunk_ids,
                retrieved_chunks=retrieved_chunks,
                raw_llm_response="",
            )
        )

    if not classifications:
        raise ValueError("No valid classification rows found in input CSV.")

    print(f"[judge-gen] Loaded {len(classifications)} classifications")
    judge_kwargs: dict = {"max_new_tokens": args.max_new_tokens}
    if args.model:
        judge_kwargs["model_name"] = args.model

    judge = LLMJudge(**judge_kwargs)
    judge_results = judge.evaluate_batch(classifications)
    save_judge_results_csv(judge_results, args.output)
    print(f"[judge-gen] Saved judge CSV -> {args.output}")


if __name__ == "__main__":
    main()
