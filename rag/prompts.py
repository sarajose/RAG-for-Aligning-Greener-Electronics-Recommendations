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
  "justification": "<structured analysis with four labelled sections — see JUSTIFICATION STRUCTURE>",
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

JUSTIFICATION STRUCTURE
───────────────────────
Your justification MUST contain all four of the following labelled sections,
each answered in 2-4 sentences:

LEGAL BASIS: Name each specific article, paragraph, and directive/regulation
that applies to this recommendation. Explain precisely how each provision
maps to (or fails to map to) the recommendation's content. Quote or closely
paraphrase the operative legal text where possible.

CORRECTNESS: Assess whether the recommendation is factually accurate and
legally sound as written. Point out any inaccuracies, overgeneralizations,
incorrect legal references, or misleading framing relative to what the
legislation actually says.

GAPS: Identify what the recommendation is missing, under-specifies, or
leaves ambiguous. Consider: thresholds not mentioned, product categories
excluded, obligations that exist in law but are absent from the recommendation,
or conditions the recommendation ignores.

IMPROVEMENTS: Give 2-3 concrete, actionable suggestions for how the
recommendation could be revised to be more complete, accurate, or better
aligned with the legislative evidence. Reference specific provisions or
legal concepts where possible.

RULES
─────
• Base your answer ONLY on the provided evidence chunks.
• If none of the chunks are relevant, choose "No explicit legal basis".
• Always cite at least one chunk_id when a legal basis exists.
• All four sections (LEGAL BASIS, CORRECTNESS, GAPS, IMPROVEMENTS) are
  mandatory — do not omit any section even if the recommendation is Aligned.
• Write in plain English prose within the justification string; use the
  section labels literally as shown above.

EXAMPLES
────────
Example 1 — Aligned
Recommendation: "Electrical and electronic equipment placed on the market must not
contain lead, mercury, cadmium, hexavalent chromium, PBB or PBDE above the maximum
concentration values."
Output:
{"label": "Aligned", "justification": "LEGAL BASIS: Article 4(1) of the RoHS Directive (2011/65/EU) directly prohibits placing EEE on the market containing lead, mercury, cadmium, hexavalent chromium, PBB or PBDE above the maximum concentration values set in Annex II. The provision is binding for all Member States and covers virtually all electrical and electronic equipment placed on the EU market. CORRECTNESS: The recommendation is factually accurate and mirrors the statutory text almost verbatim; the six substances named and the reference to concentration thresholds are consistent with the operative RoHS obligation. No legal inaccuracies are present. GAPS: The recommendation omits the Annex II concentration thresholds (e.g., 0.1 wt% for most substances, 0.01 wt% for cadmium), which are essential for practical compliance. It also does not mention the exemptions in Annex III and IV that allow certain applications to exceed these limits, nor the obligations on economic operators beyond manufacturers (importers, distributors). IMPROVEMENTS: (1) Specify the maximum concentration values from Annex II to make the recommendation self-contained and actionable. (2) Add a clause acknowledging that Annex III/IV exemptions exist so readers are not misled about absolute prohibitions. (3) Extend the scope to cover importers and distributors, not only manufacturers, in line with Article 4(2).", "cited_chunk_ids": ["<chunk_id>"]}

Example 2 — No explicit legal basis
Recommendation: "Manufacturers should publish annual material composition reports broken
down by product line, including the percentage weight of each polymer type used."
Output:
{"label": "No explicit legal basis", "justification": "LEGAL BASIS: None of the retrieved provisions establishes a binding requirement to publish annual product-line-level polymer composition reports. ESPR (Regulation 2024/1781) enables delegated acts to impose product-specific information requirements via the Digital Product Passport, and CSRD (Directive 2022/2464) mandates sustainability disclosures at the company level, but neither mandates the specific granularity described. CORRECTNESS: The recommendation is well-intentioned but presupposes a legal obligation that does not yet exist under current binding EU law. Framing it as a 'should' is appropriate given the gap, but it may mislead readers into thinking such reporting is forthcoming under a specific instrument. GAPS: There is no adopted delegated act under ESPR that specifies polymer-level reporting for consumer electronics; the CSRD disclosure framework operates at entity level, not product-line level. The recommendation also does not address to whom the report would be submitted or made public. IMPROVEMENTS: (1) Reframe the recommendation as a voluntary best practice or anticipatory measure pending future ESPR delegated acts. (2) Specify a plausible legal vehicle — e.g., future ESPR product group regulations — under which this obligation could be introduced. (3) Clarify the reporting channel (e.g., Digital Product Passport, public registry) to make the proposal more concrete.", "cited_chunk_ids": []}

Example 3 — Beyond compliance
Recommendation: "All new consumer electronics should contain a minimum of 50% post-consumer
recycled plastic in their casings."
Output:
{"label": "Beyond compliance", "justification": "LEGAL BASIS: ESPR (Regulation 2024/1781, Article 4 and Annex I) empowers the Commission to adopt delegated acts setting recycled content requirements for specific product groups via ecodesign regulations. No adopted delegated act currently mandates any recycled plastic content threshold for consumer electronics casings. The Battery Regulation (2023/1542) sets recycled content targets for batteries but does not extend to device casings. CORRECTNESS: The recommendation is technically valid as a sustainability objective but overstates current mandatory requirements. Presenting it without the qualifier that no binding threshold currently exists risks non-compliance confusion. GAPS: The recommendation does not specify which product categories within 'consumer electronics' it targets, ignores the delegated-act process required to translate ESPR principles into binding thresholds, and omits practical constraints such as material availability and verification mechanisms. IMPROVEMENTS: (1) Add explicit acknowledgement that the 50% threshold exceeds current mandatory requirements and anticipates future ESPR delegated acts. (2) Narrow the scope to a specific product category (e.g., smartphones, laptops) where recycled plastics are technically feasible. (3) Reference EN standards or the EU Ecolabel criteria that already suggest recycled content targets, providing a voluntary benchmark until binding rules apply.", "cited_chunk_ids": ["<chunk_id>"]}
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
  "reasoning": "<4-6 sentences: one sentence assessing each criterion A-D, then an overall conclusion>"
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
