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
EU regulatory framework and provide a detailed, structured analysis.

OUTPUT FORMAT (strict JSON — no markdown fences, no extra keys)
───────────────────────────────────────────────────────────────
{
  "label": "<one of the labels below>",
  "justification": "<PLAIN TEXT STRING — the four section labels written inline, NOT a nested JSON object>",
  "cited_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"]
}

CRITICAL: "justification" must be a single flat string value.
Do NOT write it as a nested object like {"LEGAL BASIS": "...", "CORRECTNESS": "..."}.
Write the section labels literally inside the string, e.g.:
"justification": "LEGAL BASIS: ... CORRECTNESS: ... GAPS: ... IMPROVEMENTS: ..."

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

JUSTIFICATION STRUCTURE
───────────────────────
Your justification MUST contain all four of the following labelled sections,
each answered in 1-2 concise sentences:

LEGAL BASIS: Name the specific article, paragraph, and directive/regulation
that applies. Explain precisely how each provision maps to the recommendation.

CORRECTNESS: Assess whether the recommendation is factually accurate and
legally sound. Note any inaccuracies or overgeneralizations.

GAPS: Identify the key missing elements: thresholds not mentioned, product
categories excluded, or obligations absent from the recommendation.

IMPROVEMENTS: Give 2 concrete, actionable suggestions for how the
recommendation could be revised. Reference specific provisions where possible.

RULES
─────
• Base your answer ONLY on the provided evidence chunks.
• If none of the chunks are relevant, choose "No explicit legal basis".
• Always cite at least one chunk_id when a legal basis exists.
• All four sections (LEGAL BASIS, CORRECTNESS, GAPS, IMPROVEMENTS) are
  mandatory — do not omit any section even if the recommendation is Aligned.
• The "justification" value must be a plain string — never a nested JSON
  object. Write LEGAL BASIS, CORRECTNESS, GAPS, IMPROVEMENTS inline.
• Be concise — the entire JSON response must complete within 512 tokens.

EXAMPLES
────────
Example 1 — Aligned
Recommendation: "Electrical and electronic equipment placed on the market must not
contain lead, mercury, cadmium, hexavalent chromium, PBB or PBDE above the maximum
concentration values."
Output:
{"label": "Aligned", "justification": "LEGAL BASIS: Article 4(1) of RoHS Directive (2011/65/EU) directly prohibits EEE containing lead, mercury, cadmium, hexavalent chromium, PBB or PBDE above maximum concentration values in Annex II. CORRECTNESS: The recommendation accurately mirrors the statutory text with no legal inaccuracies. GAPS: The Annex II concentration thresholds (e.g., 0.1 wt%) and Annex III/IV exemptions are not mentioned. IMPROVEMENTS: (1) Specify the Annex II thresholds to make the recommendation actionable. (2) Note Annex III/IV exemptions to avoid implying absolute prohibitions.", "cited_chunk_ids": ["<chunk_id>"]}

Example 2 — No explicit legal basis
Recommendation: "Manufacturers should publish annual material composition reports broken
down by product line, including the percentage weight of each polymer type used."
Output:
{"label": "No explicit legal basis", "justification": "LEGAL BASIS: No retrieved provision requires annual product-line-level polymer reports; ESPR (Regulation 2024/1781) enables such requirements via delegated acts, but none has been adopted for consumer electronics. CORRECTNESS: The recommendation presupposes an obligation that does not yet exist; the 'should' framing is appropriate but may mislead readers. GAPS: No ESPR delegated act specifies polymer-level reporting, and CSRD (Directive 2022/2464) operates at entity level, not product-line level. IMPROVEMENTS: (1) Reframe as a voluntary best practice pending future ESPR delegated acts. (2) Specify the reporting channel, such as the Digital Product Passport.", "cited_chunk_ids": []}

Example 3 — Beyond compliance
Recommendation: "All new consumer electronics should contain a minimum of 50% post-consumer
recycled plastic in their casings."
Output:
{"label": "Beyond compliance", "justification": "LEGAL BASIS: ESPR (Regulation 2024/1781, Article 4) empowers delegated acts to set recycled content requirements, but no such act currently mandates a threshold for consumer electronics casings. CORRECTNESS: The recommendation is a valid sustainability goal but overstates current requirements; no 50% threshold exists in binding EU law. GAPS: The recommendation omits specifying target product categories and ignores the delegated-act process required under ESPR. IMPROVEMENTS: (1) Acknowledge that the 50% threshold exceeds current requirements and anticipates future ESPR delegated acts. (2) Narrow scope to a specific product category where recycled plastics are technically feasible.", "cited_chunk_ids": ["<chunk_id>"]}
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
   cite specific articles and provisions by name, logically explain the
   label, and correctly interpret the legislative text?
   5 = precise citations with sound legal reasoning, 1 = vague or fabricated.

C. **Evidence usage** (1-5): Are the cited chunk IDs relevant and
   sufficient to support the classification?
   5 = all key relevant chunks cited, 1 = no relevant citations or wrong chunks.

D. **Completeness** (1-5): Does the justification identify legislative
   gaps in the recommendation (what is missing or under-specified) AND
   provide concrete suggestions for how the recommendation could be improved?
   5 = detailed gap analysis with actionable improvements citing specific
   provisions, 1 = no mention of gaps or improvement potential.

OUTPUT FORMAT (strict JSON — no markdown fences)
────────────────────────────────────────────────
{
  "label_score": <int 1-5>,
  "justification_score": <int 1-5>,
  "evidence_score": <int 1-5>,
  "completeness_score": <int 1-5>,
  "overall_score": <float — average of the four scores>,
  "reasoning": "<2-3 sentences: briefly assess the key strengths and weaknesses across criteria A-D>"
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

Evaluate the classification on all four criteria (A: label correctness,
B: justification quality, C: evidence usage, D: completeness / gap analysis).
Respond with the JSON object only.
"""



# FORMATTING HELPERS
def format_evidence_block(chunks: list[Chunk], max_chars_per_chunk: int | None = 600) -> str:
    """Format retrieved chunks into a numbered evidence block.

    When a chunk carries ``article_text`` (parent-child chunking), the full
    article text is used as the body; otherwise ``chunk.text`` is used.

    Parameters
    ----------
    max_chars_per_chunk : int | None
        Truncate each chunk body to this many characters before building the
        prompt.  Keeps prompts within the 4096-token input limit on consumer
        GPUs (600 chars × 10 chunks ≈ 1 500 tokens, well inside budget).
        Pass ``None`` only on high-VRAM machines where you want the full text.
    """
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] id={c.id} | {c.document} | {c.article}"
        if c.article_subtitle:
            header += f" — {c.article_subtitle}"
        if c.paragraph:
            header += f" | §{c.paragraph}"
        body = c.article_text if c.article_text else c.text
        if max_chars_per_chunk is not None and len(body) > max_chars_per_chunk:
            body = body[:max_chars_per_chunk] + "…"
        parts.append(f"{header}\n{body}\n")
    return "\n".join(parts)


def build_classifier_messages(
    recommendation: str,
    chunks: list[Chunk],
    max_chars_per_chunk: int | None = 600,
) -> list[dict[str, str]]:
    """Build the chat messages for the alignment classifier."""
    evidence_block = format_evidence_block(chunks, max_chars_per_chunk=max_chars_per_chunk)
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
{"label_score": <int 1-5>, "justification_score": <int 1-5>, "evidence_score": <int 1-5>, "completeness_score": <int 1-5>, "overall_score": <float>, "reasoning": "<one English sentence>"}
"""

_JUDGE_RETRY_USER = """\
Score the following AI classification on a 1-5 scale for each dimension:
label_score (label correctness), justification_score (citation quality),
evidence_score (relevant chunks cited), completeness_score (gaps and
improvement suggestions present).

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
    max_chars_per_chunk: int | None = 600,
) -> list[dict[str, str]]:
    """Build the chat messages for the LLM-as-judge."""
    evidence_block = format_evidence_block(chunks, max_chars_per_chunk=max_chars_per_chunk)
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
