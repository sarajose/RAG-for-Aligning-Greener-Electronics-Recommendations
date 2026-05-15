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
  "cited_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"],
  "justification": "<PLAIN TEXT STRING — the four section labels written inline, NOT a nested JSON object>"
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
Your justification MUST contain all four of the following labelled sections.
Write each section as a structured paragraph of 3-5 sentences that includes:
  • The exact article number and paragraph (e.g. "Article 4(1)")
  • The full directive or regulation name and number
  • A verbatim quote or close paraphrase of the operative legislative text
  • Specific numerical thresholds with units ONLY when the exact number appears
    in the retrieved evidence chunks — never infer or approximate a threshold
    that is not stated in the evidence
  • Recital references where they clarify legislative intent

LEGAL BASIS: Identify every provision that maps to the recommendation.
Quote or closely paraphrase the operative text. If multiple provisions
apply, address each one in turn.

CORRECTNESS: Assess factual and legal accuracy. Note any
overgeneralisation, missing thresholds, incorrect scope claims, or
conflation of prospective and currently-in-force obligations.

GAPS: Identify every specific element absent from the recommendation:
missing concentration thresholds, excluded product categories, exemption
regimes not mentioned, delegated acts not yet adopted, or obligations
not yet in force.

IMPROVEMENTS: Give 2-3 concrete revision suggestions, each referencing a
specific provision. Phrase them as actionable edits, for example: "add
'above 0.1 wt% per homogeneous material' per Annex II entry 1 of
2011/65/EU to match the operative legal text."

RULES
─────
• Base your answer ONLY on the provided evidence chunks.
• If none of the chunks are relevant, choose "No explicit legal basis".
• Always cite at least one chunk_id when a legal basis exists.
• Your cited_chunk_ids MUST be exact hexadecimal IDs strings (e.g., "5046c808" or "d18e4f05") exactly as they appear after "id=" in the evidence header. Do NOT use array indices (e.g. "2"), paragraph numbers, or partial strings (e.g. "c808").
• Both binding legal obligations (Regulations/Directives) and policy documents (Communications, Action Plans, Whitepapers) can support an "Aligned" label. However, if the alignment is based solely on policy or recommendation documents rather than binding law, you MUST explicitly state this distinction in the LEGAL BASIS and CORRECTNESS sections of your justification.
• All four sections (LEGAL BASIS, CORRECTNESS, GAPS, IMPROVEMENTS) are
  mandatory — do not omit any section even if the recommendation is Aligned.
• The "justification" value must be a plain string — never a nested JSON
  object. Write LEGAL BASIS, CORRECTNESS, GAPS, IMPROVEMENTS inline.
• NEVER cite a numerical threshold unless the exact value appears verbatim
  in the retrieved evidence text. If no threshold is present in the chunks,
  do not mention one.

EXAMPLES
────────
Example 1 — Aligned
Recommendation: "Electrical and electronic equipment placed on the market must not
contain lead, mercury, cadmium, hexavalent chromium, PBB or PBDE above the maximum
concentration values."
Output:
{"label": "Aligned", "justification": "LEGAL BASIS: Article 4(1) of RoHS Directive 2011/65/EU directly prohibits placing EEE on the market if it contains lead, mercury, cadmium, hexavalent chromium, PBB, or PBDE 'in concentrations exceeding the maximum concentration values laid down in Annex II.' Annex II sets thresholds of 0.1% by weight per homogeneous material for lead (entry 1), mercury (entry 2), hexavalent chromium (entry 3), PBB (entry 4), and PBDE (entry 5), and 0.01% for cadmium (entry 6). The prohibition applies to all EEE within the scope of Article 2(1), subject to Annex III and IV exemptions for critical-use applications. Recital 14 confirms these values reflect the precautionary principle and best available substitution technology. CORRECTNESS: The recommendation accurately mirrors the operative text of Article 4(1) and correctly names all six restricted substances. It contains no factual inaccuracies; however, it omits the Annex II concentration thresholds (0.1 wt% and 0.01 wt% for cadmium), which are the operative compliance criteria — a manufacturer relying on this recommendation alone cannot determine the actual legal limit. GAPS: The Annex II numerical thresholds are absent, making the recommendation non-actionable for compliance purposes. The Annex III and IV exemption regimes are not mentioned, implying an absolute prohibition broader than the law provides — for example, Annex III exemption 6(a) permits lead in high-melting-point solders above 85% lead content. The product-scope exclusions of Article 2(2) (e.g., military equipment, fixed industrial plants) are also unaddressed. IMPROVEMENTS: (1) Add the operative threshold language: 'above the maximum concentration values in Annex II (0.1% by weight per homogeneous material for lead, mercury, Cr(VI), PBB, and PBDE; 0.01% for cadmium)' to make the recommendation directly actionable for compliance purposes. (2) Cross-reference Annex III and IV exemptions and note that validity periods are subject to review under Article 5(1), so manufacturers in affected sectors should consult the Commission's current exemption register.", "cited_chunk_ids": ["a3f9b2c7", "d18e4f05"]}

Example 2 — No explicit legal basis
Recommendation: "Manufacturers should publish annual material composition reports broken
down by product line, including the percentage weight of each polymer type used."
Output:
{"label": "No explicit legal basis", "justification": "LEGAL BASIS: No retrieved provision currently requires manufacturers to publish annual product-line-level polymer composition reports. ESPR Regulation 2024/1781 Article 4(1)(d) empowers the Commission to impose information-disclosure requirements through delegated acts, and Article 9 establishes the Digital Product Passport as a vehicle for such disclosures, but no delegated act for consumer electronics specifying polymer-level reporting has been adopted. CSRD Directive 2022/2464 Article 19a and ESRS E5 require material-flow disclosure at entity level, not at product-line granularity. Recital 26 of ESPR confirms that delegated acts must be preceded by a preparatory study, making such requirements prospective rather than current. CORRECTNESS: The recommendation presupposes an obligation that does not yet exist in binding EU law; the 'should' framing is appropriate for a voluntary best practice but may mislead readers into believing such reporting is legally required. Entity-level CSRD disclosures and the prospective ESPR delegated-act framework are distinct instruments that cannot satisfy the product-line granularity described. GAPS: No ESPR delegated act specifies polymer-level reporting or the product-line as a reporting unit. The recommendation does not identify the reporting channel, the reference unit for 'percentage weight,' or whether the obligation would apply to primary or recycled polymer content. The CSRD entity-level framework would not satisfy this product-line requirement even if mandated. IMPROVEMENTS: (1) Reframe as a forward-looking voluntary best practice: 'Manufacturers are encouraged to publish polymer-level material composition data at product-line granularity in anticipation of future ESPR Article 4(1)(d) requirements and in preparation for Digital Product Passport integration under Article 9.' (2) Specify the reporting channel — recommend the Digital Product Passport under ESPR Article 9 as the publication medium — and define the reference unit as 'percentage weight per homogeneous material per product category.'", "cited_chunk_ids": []}

Example 3 — Beyond compliance
Recommendation: "All new consumer electronics should contain a minimum of 50% post-consumer
recycled plastic in their casings."
Output:
{"label": "Beyond compliance", "justification": "LEGAL BASIS: ESPR Regulation 2024/1781 Article 4(1)(b) and (c) empowers the Commission to set minimum recycled-content requirements through delegated acts, and Article 5(1) lists 'use of recycled content' as a product parameter on which ecodesign requirements may be imposed. No delegated act adopted under ESPR currently sets any recycled-content threshold for consumer electronics casings; the Working Plan 2022-2024 does not include a consumer electronics delegated act specifying a 50% recycled plastic threshold. The existing Ecodesign Regulation (EU) 2019/1782 for consumer electronics addresses energy efficiency but not material composition. No retrieved provision mandates any recycled plastic content threshold for consumer electronics. CORRECTNESS: The 50% threshold is a valid sustainability ambition but significantly overstates current EU legal requirements. The recommendation correctly signals the direction of future ESPR delegated acts but incorrectly implies an existing or imminent obligation; the phrase 'should contain' without clear qualification between regulatory aspiration and current law is misleading. GAPS: The recommendation omits product-category specificity — televisions, smartphones, and laptops each have distinct material and structural constraints. It does not address the delegated-act process under ESPR Article 4, technical feasibility barriers such as flame-retardant certification requirements, or the Ecodesign Consultation Forum process under Article 18. IMPROVEMENTS: (1) Qualify the recommendation to accurately characterise its regulatory status: 'Beyond current EU requirements, manufacturers are encouraged to target a minimum of 30-50% post-consumer recycled plastic in casings where technically feasible, anticipating future ESPR delegated acts under Article 4(1)(b).' (2) Narrow scope to a specific product category where feasibility studies already demonstrate high recycled-content viability, referencing product-category preparatory studies published under the ESPR Working Plan.", "cited_chunk_ids": ["b7c2e918"]}

Example 4 — Conditional
Recommendation: "All consumer electronics placed on the EU market must be accompanied by
a Digital Product Passport containing material composition data, a repairability score,
and end-of-life instructions before market placement."
Output:
{"label": "Conditional", "justification": "LEGAL BASIS: ESPR Regulation 2024/1781 Article 9(1) establishes the Digital Product Passport (DPP) as a mandatory information vehicle that 'shall accompany products or part of the information shall be accessible via a data carrier.' Article 4(1)(h) lists the DPP as a product requirement imposable through delegated acts, and Article 7 specifies minimum DPP content including material composition (Article 7(5)(a)) and end-of-life information (Article 7(5)(g)). However, DPP requirements only become operative for a product group once a delegated act under Article 4 enters into force for that group; no such act has been adopted for general consumer electronics as of the current date. CORRECTNESS: The recommendation accurately identifies the DPP as the correct ESPR instrument and correctly describes content elements drawn from Article 7. However, it incorrectly characterises a conditional future obligation as a current 'must' requirement; DPP obligations for consumer electronics are contingent on a forthcoming delegated act and are not yet enforceable. GAPS: The recommendation does not identify the delegated-act conditionality that makes this obligation prospective. It omits the phased implementation timeline (ESPR Working Plan 2022-2024 prioritises batteries and textiles before consumer electronics), the harmonised repairability-score methodology still under development, and interoperability requirements with the European Product Registry for Energy Labelling under Article 12. IMPROVEMENTS: (1) Reframe as a conditional obligation: 'Once a delegated act under ESPR Article 4 is adopted for consumer electronics, products must be accompanied by a Digital Product Passport under Article 9(1) containing at minimum the information specified in Articles 7(5)(a) and 7(5)(g).' (2) Note that the Commission will define product-specific DPP data schemas through implementing acts under Article 9(5) and recommend that manufacturers engage with the Ecodesign Consultation Forum under Article 18 to influence schema development for their product categories.", "cited_chunk_ids": ["c94f1a83", "d27b3c60"]}
"""

CLASSIFIER_USER_TEMPLATE = """\
{context_block}RECOMMENDATION
──────────────
{recommendation}

RETRIEVED EVIDENCE (top-{k} chunks)
────────────────────────────────────
{evidence_block}

Respond with the JSON object only.
"""

_CONTEXT_BLOCK_TEMPLATE = """\
CONTEXT (background only — do not evaluate this)
─────────────────────────────────────────────────
This recommendation is part of the group titled: "{title}"
Use this title only to understand what the recommendation below refers to.
Evaluate only the RECOMMENDATION text, not the title.

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

GROUNDING STEP (complete this before scoring)
─────────────────────────────────────────────
Read the retrieved evidence chunks carefully. Identify:
  • Which specific article numbers and paragraphs actually appear in the chunks.
  • Which verbatim phrases or numerical thresholds (with exact values) appear.
  • Which documents are binding law (regulation/directive) vs. non-binding
    guidance (commission recommendation, communication, strategy).
Use this inventory to verify every claim in the classifier's justification.
Do NOT credit an article, quote, or threshold unless you can locate it in the
retrieved evidence above.

EVALUATION CRITERIA
───────────────────
A. **Label correctness** (1-5): Is the predicted label appropriate given
   the evidence?  5 = perfect fit, 1 = completely wrong label.

B. **Justification quality** (1-5): Does the justification accurately cite
   articles and provisions that are actually present in the retrieved evidence?
   Reward: citations you can locate verbatim or near-verbatim in the chunks;
   article-paragraph precision (e.g. "Article 4(1)"); numerical thresholds
   with units ONLY when those exact numbers appear in the retrieved chunks.
   Penalise: any article, quote, or threshold the classifier mentions that you
   cannot find in the retrieved evidence; vague references like "EU law
   states..." without a provision number; claiming thresholds exist when none
   appear in the chunks.
   5 = all citations traceable to the evidence, sound legal reasoning;
   1 = citations fabricated or absent from the evidence.

C. **Evidence usage** (1-5): Are the cited chunk IDs relevant and
   sufficient to support the classification?
   Consider whether the strongest available chunks were used — note if binding
   law chunks were ignored in favour of weaker guidance chunks, or vice versa.
   5 = all key relevant chunks cited, 1 = no relevant citations or wrong chunks.

D. **Completeness** (1-5): Does the justification identify legislative
   gaps in the recommendation (what is missing or under-specified) AND
   provide concrete suggestions for how the recommendation could be improved?
   5 = detailed gap analysis with actionable improvements citing specific
   provisions, 1 = no mention of gaps or improvement potential.

SCORE ANCHORS (to prevent score clustering)
────────────────────────────────────────────
• Score 5: Every cited article, quote, and threshold is verifiable in the
  retrieved evidence. No fabrication detected. Reasoning is precise and legally
  sound.
• Score 4: Minor imprecision (e.g. slightly wrong paragraph number) but no
  fabrication and the core legal mapping is correct.
• Score 3: One or two citations are vague or not traceable to the evidence, OR
  the label is technically defensible but a better label exists.
• Score 2: Key citations are fabricated or wrong, OR the label is applied
  inconsistently with the justification text.
• Score 1: Systematic fabrication, completely wrong label, or no legal
  grounding at all.
Reserve score 5 for genuinely exceptional outputs. If a justification claims
thresholds or quotes absent from the retrieved chunks, justification_score
MUST be ≤ 3.

CONSISTENCY RULE
────────────────
If in criterion A you conclude the label is wrong or questionable,
label_score MUST be ≤ 3. A score of 4–5 certifies you agree with the label.
Do not write contradictory reasoning alongside a high label_score.

OUTPUT FORMAT (strict JSON — no markdown fences)
────────────────────────────────────────────────
{
  "label_score": <int 1-5>,
  "justification_score": <int 1-5>,
  "evidence_score": <int 1-5>,
  "completeness_score": <int 1-5>,
  "overall_score": <float — average of the four scores>,
  "reasoning": "<Detailed assessment: for each criterion (A-D) state (1) what the retrieved evidence actually contains, (2) what the classifier claimed, (3) whether they match. Do NOT credit a threshold or quote unless you can locate it in the retrieved evidence above.>"
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
def format_evidence_block(chunks: list[Chunk], max_chars_per_chunk: int | None = None) -> str:
    """Format retrieved chunks into a numbered evidence block.

    When a chunk carries ``article_text`` (parent-child chunking), the full
    article text is used as the body; otherwise ``chunk.text`` is used.

    Parameters
    ----------
    max_chars_per_chunk : int | None
        Truncate each chunk body to this many characters before building the
        prompt.  Defaults to ``None`` (no truncation) for 24+ GiB GPUs where
        full article text fits comfortably within a 16k token context window.
        Set to e.g. 600 for constrained environments (4 GiB GPU).
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
    max_chars_per_chunk: int | None = None,
    title: str = "",
) -> list[dict[str, str]]:
    """Build the chat messages for the alignment classifier."""
    evidence_block = format_evidence_block(chunks, max_chars_per_chunk=max_chars_per_chunk)
    context_block = _CONTEXT_BLOCK_TEMPLATE.format(title=title) if title.strip() else ""
    user_msg = CLASSIFIER_USER_TEMPLATE.format(
        context_block=context_block,
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
RECOMMENDATION
──────────────
{recommendation}

RETRIEVED EVIDENCE (abbreviated)
─────────────────────────────────
{evidence_snippet}

ASSISTANT'S CLASSIFICATION
──────────────────────────
Label: {label}
Justification: {justification}

Score on label correctness (label_score), citation quality vs. the evidence
above (justification_score), evidence relevance (evidence_score), and gap
analysis (completeness_score). Reply with JSON only.
"""


def build_judge_retry_messages(
    label: str,
    justification: str,
    recommendation: str = "",
    chunks: "list | None" = None,
) -> list[dict[str, str]]:
    """Fallback prompt used when the primary judge parse fails.

    Includes the recommendation and a short evidence snippet so that
    evidence_score can be grounded in the actual retrieved chunks rather
    than scored blind from the justification text alone.
    """
    if chunks:
        snippet = format_evidence_block(chunks, max_chars_per_chunk=300)[:1200]
    else:
        snippet = "(no evidence available)"
    return [
        {"role": "system", "content": _JUDGE_RETRY_SYSTEM},
        {"role": "user", "content": _JUDGE_RETRY_USER.format(
            recommendation=recommendation[:300],
            evidence_snippet=snippet,
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
    max_chars_per_chunk: int | None = None,
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
