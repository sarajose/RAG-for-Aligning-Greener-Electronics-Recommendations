import re
import pdfplumber
import pandas as pd
from hashlib import sha1

POLICY_REC_RE = re.compile(r'Policy recommendation\s+(\w+)\s*:\s*(.*)', re.IGNORECASE)
BULLET_RE = re.compile(r'^\s*[•\-\*]\s+(.*)')

def read_pages(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            yield i, page.extract_text() or ""

def stable_id(*parts: str) -> str:
    s = "||".join(parts).encode("utf-8")
    return sha1(s).hexdigest()[:12]

def extract_policy_recommendations(pages):
    """Extract blocks that start with 'Policy recommendation X:'."""
    rows = []
    buffer = None  # (label, page_start, text_parts)
    for page_no, text in pages:
        lines = text.splitlines()
        for line in lines:
            m = POLICY_REC_RE.search(line)
            if m:
                # flush previous
                if buffer:
                    label, p0, parts = buffer
                    full = " ".join(parts).strip()
                    rows.append((label, p0, page_no, full))
                label = f"Policy recommendation {m.group(1)}"
                first = m.group(2).strip()
                buffer = (label, page_no, [first] if first else [])
            else:
                if buffer:
                    buffer[2].append(line.strip())
    # flush final
    if buffer:
        label, p0, parts = buffer
        full = " ".join(parts).strip()
        rows.append((label, p0, p0, full))
    return rows

def extract_bullets(pages):
    """Extract bullet items; keeps page numbers."""
    rows = []
    current = None  # (page_start, parts)
    for page_no, text in pages:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = BULLET_RE.match(line)
            if m:
                # flush previous bullet
                if current:
                    p0, parts = current
                    rows.append((p0, page_no, " ".join(parts).strip()))
                current = (page_no, [m.group(1).strip()])
            else:
                if current:
                    current[1].append(line)
    if current:
        p0, parts = current
        rows.append((p0, p0, " ".join(parts).strip()))
    return rows

def build_recommendation_dataset(pdf_paths):
    out = []
    for path in pdf_paths:
        pages = list(read_pages(path))

        # 1) Policy recommendation blocks (if present)
        pr = extract_policy_recommendations(pages)
        for label, p0, p1, txt in pr:
            rid = stable_id(path, str(p0), label, txt[:60])
            out.append({
                "rec_id": rid,
                "recommendation_text": re.sub(r"\s+", " ", txt).strip(),
                "source_doc": path,
                "page_start": p0,
                "page_end": p1,
                "section_path": label,
                "extraction_method": "policy_recommendation_regex",
            })

        # Bullets (good for Orgalim and Higher Education docs)
        bullets = extract_bullets(pages)
        for p0, p1, txt in bullets:
            if len(txt) < 25:
                continue
            rid = stable_id(path, str(p0), "bullet", txt[:60])
            out.append({
                "rec_id": rid,
                "recommendation_text": re.sub(r"\s+", " ", txt).strip(),
                "source_doc": path,
                "page_start": p0,
                "page_end": p1,
                "section_path": "",
                "extraction_method": "bullet",
            })

    return pd.DataFrame(out)

# Example usage:
pdfs = [
    "Orgalim key recommendations on digital policy.pdf",
    "Policy Recommendations for Electronic Public Procurement.pdf",
    "Policy Recommendations for Higher Education Institutions to Begin Advancing from Digital Transformation to Bifurcation.pdf",
]
df = build_recommendation_dataset(pdfs)
df.to_csv("recommendations_dataset.csv", index=False)
print(df.head())
