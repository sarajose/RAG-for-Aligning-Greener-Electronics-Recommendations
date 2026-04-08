"""
Prompt templates for the alignment classifier and LLM-as-judge.

Kept in a single file so that all prompt engineering is co-located
and easy to iterate on.
"""

from config import ALIGNMENT_LABELS
from data_models import Chunk

# ALIGNMENT CLASSIFIER PROMPTS
CLASSIFIER_SYSTEM_PROMPT = """\
You are a legal-policy analyst specialising in EU sustainability regulation.

TASK
────
Given a sustainability recommendation and a set of retrieved EU legal
provisions, determine how the recommendation aligns with the current
EU regulatory framework.

OUTPUT FORMAT (strict JSON — no markdown fences, no extra keys)
───────────────────────────────────────────────────────────────
{
  "label": "<one of the labels below>",
  "justification": "<2-4 sentences citing specific provisions>",
  "cited_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"]
}

ALIGNMENT LABELS
────────────────
1. Aligned
   The recommendation is directly supported by or consistent with
   existing EU legislation.

2. Conditional
   Partial alignment — the recommendation depends on delegated /
   implementing acts, thresholds, or conditions not yet fully specified.

3. Conflicting
   The recommendation contradicts or is incompatible with provisions
   in the current legal framework.

4. No explicit legal basis
   No retrieved provision specifically addresses the recommendation;
   there is a regulatory gap.

5. Beyond compliance
   The recommendation exceeds what legislation requires, proposing
   stricter or additional measures.

RULES
─────
• Base your answer ONLY on the provided evidence chunks.
• If none of the chunks are relevant, choose "No explicit legal basis".
• Always cite at least one chunk_id when a legal basis exists.
• Keep the justification concise and factual.

EXAMPLES
────────
Example 1 — Aligned
Recommendation: "Electrical and electronic equipment placed on the market must not
contain lead, mercury, cadmium, hexavalent chromium, PBB or PBDE above the maximum
concentration values."
Output:
{"label": "Aligned", "justification": "Article 4(1) of the RoHS Directive (2011/65/EU) directly prohibits placing EEE on the market that contains restricted substances exceeding the maximum concentration values listed in Annex II. The recommendation maps precisely to this binding obligation.", "cited_chunk_ids": ["<chunk_id>"]}

Example 2 — No explicit legal basis
Recommendation: "Manufacturers should publish annual material composition reports broken
down by product line, including the percentage weight of each polymer type used."
Output:
{"label": "No explicit legal basis", "justification": "None of the retrieved provisions require product-line-level annual polymer composition reporting. ESPR and CSRD address reporting at a higher level of aggregation and do not mandate this specific granularity. No chunk provides explicit grounding for this obligation.", "cited_chunk_ids": []}

Example 3 — Beyond compliance
Recommendation: "All new consumer electronics should contain a minimum of 50% post-consumer
recycled plastic in their casings."
Output:
{"label": "Beyond compliance", "justification": "ESPR (Regulation 2024/1781) enables delegated acts to impose recycled content requirements, but no adopted delegated act currently mandates a 50% post-consumer recycled plastic threshold for consumer electronics casings. The recommendation exceeds what current binding obligations require.", "cited_chunk_ids": ["<chunk_id>"]}
"""

CLASSIFIER_USER_TEMPLATE = """\
RECOMMENDATION
──────────────
{recommendation}

RETRIEVED EVIDENCE (top-{k} chunks)
────────────────────────────────────
{evidence_block}

Respond with the JSON object only.
"""


# LLM-AS-JUDGE PROMPTS

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of an AI assistant's
alignment classification of sustainability recommendations against EU
legislation.

You will be given:
1. A sustainability recommendation.
2. Retrieved legal evidence chunks.
3. The assistant's classification (label + justification).

EVALUATION CRITERIA
───────────────────
A. **Label correctness** (1-5): Is the predicted label appropriate given
   the evidence?  5 = perfect, 1 = completely wrong.

B. **Justification quality** (1-5): Does the justification accurately
   cite specific provisions and logically explain the label?
   5 = thorough and well-grounded, 1 = irrelevant or fabricated.

C. **Evidence usage** (1-5): Are the cited chunk IDs relevant and
   sufficient?  5 = all relevant chunks cited, 1 = no relevant citations.

OUTPUT FORMAT (strict JSON — no markdown fences)
────────────────────────────────────────────────
{
  "label_score": <int 1-5>,
  "justification_score": <int 1-5>,
  "evidence_score": <int 1-5>,
  "overall_score": <float — average of the three>,
  "reasoning": "<1-3 sentences explaining your assessment>"
}

LANGUAGE REQUIREMENT (mandatory)
────────────────────────────────
- Write all fields in English only.
- The "reasoning" field must be English prose (no Chinese or other languages).
- If uncertain, still answer in English.
"""

JUDGE_USER_TEMPLATE = """\
RECOMMENDATION
──────────────
{recommendation}

RETRIEVED EVIDENCE (top-{k} chunks)
────────────────────────────────────
{evidence_block}

ASSISTANT'S CLASSIFICATION
──────────────────────────
Label: {label}
Justification: {justification}
Cited chunks: {cited_chunks}

Evaluate the classification and respond with the JSON object only.
"""



# FORMATTING HELPERS
def format_evidence_block(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a numbered evidence block.

    When a chunk carries ``article_text`` (parent-child chunking), the full
    article text is shown instead of the short paragraph so the classifier
    receives richer generation context.
    """
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] id={c.id} | {c.document} | {c.article}"
        if c.article_subtitle:
            header += f" — {c.article_subtitle}"
        if c.paragraph:
            header += f" | §{c.paragraph}"
        body = c.article_text if c.article_text else c.text
        parts.append(f"{header}\n{body}\n")
    return "\n".join(parts)


def build_classifier_messages(
    recommendation: str,
    chunks: list[Chunk],
) -> list[dict[str, str]]:
    """Build the chat messages for the alignment classifier."""
    evidence_block = format_evidence_block(chunks)
    user_msg = CLASSIFIER_USER_TEMPLATE.format(
        recommendation=recommendation,
        k=len(chunks),
        evidence_block=evidence_block,
    )
    return [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


_JUDGE_RETRY_SYSTEM = """\
You are a scoring assistant. Reply with ONLY a JSON object — no other text.

Required format:
{"label_score": <int 1-5>, "justification_score": <int 1-5>, "evidence_score": <int 1-5>, "overall_score": <float>, "reasoning": "<one English sentence>"}
"""

_JUDGE_RETRY_USER = """\
Score the following AI classification on a 1-5 scale for each dimension.

Label assigned: {label}
Justification: {justification}

Reply with JSON only.
"""


def build_judge_retry_messages(
    label: str,
    justification: str,
) -> list[dict[str, str]]:
    """Minimal fallback prompt used when the primary judge parse fails."""
    return [
        {"role": "system", "content": _JUDGE_RETRY_SYSTEM},
        {"role": "user", "content": _JUDGE_RETRY_USER.format(
            label=label,
            justification=justification[:400],
        )},
    ]


def build_judge_messages(
    recommendation: str,
    chunks: list[Chunk],
    label: str,
    justification: str,
    cited_chunk_ids: list[str],
) -> list[dict[str, str]]:
    """Build the chat messages for the LLM-as-judge."""
    evidence_block = format_evidence_block(chunks)
    user_msg = JUDGE_USER_TEMPLATE.format(
        recommendation=recommendation,
        k=len(chunks),
        evidence_block=evidence_block,
        label=label,
        justification=justification,
        cited_chunks=", ".join(cited_chunk_ids) if cited_chunk_ids else "none",
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
