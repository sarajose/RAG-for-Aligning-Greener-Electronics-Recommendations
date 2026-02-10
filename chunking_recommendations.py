"""
Parse a structured recommendations TXT (sections → subsections → bullets)
into a CSV with one atomic recommendation per row.

Approach: regex-based structural parsing + POS-based smart sentence splitting.
Uses spaCy POS tags (not hardcoded word lists) to decide whether a sentence
introduces a new action or merely qualifies the previous one.

Usage:
  python -m spacy download en_core_web_sm
  python chunking_recommendations.py -i data/recommendations/recommendations.txt -o outputs/recommendations.csv
"""

import re, csv, argparse, spacy

# ── Load spaCy (small model, just for POS tagging) ──
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit("Run: python -m spacy download en_core_web_sm")

# ── Patterns ──
SECTION_RE = re.compile(r'^(\d+\.\d+)\.\s*(.+)')
SUBSECTION_RE = re.compile(r'^[·\t\s]*(?:Best practices|Actions needed)', re.I)
TOP_BULLET_RE = re.compile(r'^[·•\-\*§]\t?\s*')
SUB_BULLET_RE = re.compile(r'^o\s+')
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def is_new_recommendation(sent: str) -> bool:
    """Use POS tags to decide if a sentence introduces a new actionable idea.
    
    A sentence is a NEW recommendation if:
      - It starts with a VERB (imperative: "Expand...", "Introduce...")
      - It contains a modal auxiliary (should, must, shall, could...)
    A sentence is a QUALIFIER (merge with previous) if:
      - It starts with PRON or DET (This..., These..., Such..., Its...)
      - It starts with ADV/SCONJ (However, Meanwhile, Additionally...)
    """
    doc = nlp(sent)
    if len(doc) == 0:
        return False
    first = doc[0]
    # Starts with pronoun/determiner → qualifier ("This measure...", "These often...")
    if first.pos_ in ("PRON", "DET"):
        return False
    # Starts with adverb/conjunction → discourse connector ("However...", "Meanwhile...")
    if first.pos_ in ("ADV", "SCONJ", "CCONJ"):
        return False
    # Contains a modal auxiliary → new obligation/recommendation
    if any(tok.pos_ == "AUX" and tok.dep_ == "aux" and tok.tag_ == "MD" for tok in doc):
        return True
    # Starts with verb (imperative) → new action
    if first.pos_ == "VERB":
        return True
    return True  # default: treat as new


def smart_split(body: str) -> list[str]:
    """Split multi-sentence text only when a sentence introduces a new action.
    Qualifying/explanatory sentences stay merged with the preceding one."""
    sents = [s.strip() for s in SENT_RE.split(body) if s.strip()]
    if len(sents) <= 1:
        return sents
    groups: list[list[str]] = [[sents[0]]]
    for s in sents[1:]:
        if is_new_recommendation(s):
            groups.append([s])
        else:
            groups[-1].append(s)
    return [" ".join(g) for g in groups]


def parse(text: str) -> list[dict]:
    """Parse structured TXT into a flat list of recommendations.
    
    Strategy: smart sentence splitting — only split when a new sentence
    introduces its own actionable statement. Qualifiers stay merged.
    Sub-bullets get their parent's title as context.
    """
    section = subsection = ""
    # Two-pass: first collect raw bullet hierarchy, then flatten

    # ── Pass 1: collect bullets ──
    entries: list[dict] = []  # {"section","subsection","text","subs":[str]}
    current: dict | None = None
    cont_lines: list[str] = []

    def flush():
        nonlocal current, cont_lines
        if current is None:
            cont_lines = []
            return
        if cont_lines:
            current["text"] += " " + " ".join(cont_lines)
        entries.append(current)
        current = None
        cont_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        m = SECTION_RE.match(line)
        if m:
            flush()
            section = f"{m.group(1)}. {m.group(2).strip()}"
            subsection = ""
            continue

        if SUBSECTION_RE.match(line):
            flush()
            subsection = "Best practices" if "best" in line.lower() else "Actions needed"
            continue

        if not section:
            continue

        # Sub-bullet → attach to current parent
        if SUB_BULLET_RE.match(line):
            if current is not None:
                if cont_lines:
                    current["text"] += " " + " ".join(cont_lines)
                    cont_lines = []
                current["subs"].append(SUB_BULLET_RE.sub("", line).strip())
            continue

        # Top-level bullet
        if TOP_BULLET_RE.match(line):
            flush()
            current = {
                "section": section, "subsection": subsection,
                "text": TOP_BULLET_RE.sub("", line).strip(), "subs": [],
            }
            continue

        # Continuation line
        cont_lines.append(line)

    flush()

    # ── Pass 2: flatten into rows ──
    rows = []
    for e in entries:
        txt = e["text"].strip()
        # Remove editorial comments like "-> already addressed..."
        txt = re.sub(r'\s*->\s*.+$', '', txt)

        # Extract title from "Title: body" pattern
        title = ""
        body = txt
        cm = re.match(r'^([A-Z][^:]{5,80}):\s*(.+)', txt)
        if cm and len(cm.group(2)) > 30:
            title = cm.group(1).strip()
            body = cm.group(2).strip()

        # If bullet ends with ":" it's just a parent intro — skip standalone
        # but use it as prefix for sub-bullets
        is_stub = body.rstrip().endswith(":") or len(body) < 25

        if e["subs"]:
            # Parent has sub-bullets: emit each sub-bullet with parent context
            prefix = title or body.rstrip(": ")
            for sub in e["subs"]:
                rows.append({
                    "section": e["section"], "subsection": e["subsection"],
                    "title": prefix,
                    "recommendation": sub,
                })
        elif not is_stub:
            # Smart-split: separate distinct recommendations, keep qualifiers merged
            for chunk in smart_split(body):
                rows.append({
                    "section": e["section"], "subsection": e["subsection"],
                    "title": title,
                    "recommendation": chunk,
                })

    return rows


def main(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    rows = parse(text)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "subsection", "title", "recommendation"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} recommendations to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()
    main(args.input, args.output)