"""
Chunk EU legislation HTML files (EUR-Lex format) into a CSV of
paragraph-level legal provisions.

Works with any EUR-Lex consolidated-text HTML, including:
  WEEE 2012/19/EU, RoHS 2011/65/EU, ESPR 2024/1781, REACH 1907/2006,
  EU Green Deal strategy docs, CAEP, etc.

Chunking strategy
─────────────────
  1. Parse the HTML DOM with BeautifulSoup.
  2. Extract document metadata (title, short name).
  3. Walk the legal hierarchy:
       Document → Chapter/Title → Article → Paragraph → List items
  4. One chunk = one numbered paragraph (or full article if unnumbered).
     List items stay attached to their parent paragraph so that
     "shall / except / where" clauses remain together.
  5. Write a flat CSV: one provision per row with full hierarchy context.

Usage
─────
  pip install beautifulsoup4 lxml
  python chunking_evidence.py -i data/evidence -o outputs/evidence.csv
  python chunking_evidence.py -i data/evidence/WEE-08.04.2024.html -o outputs/weee.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from hashlib import sha256

from bs4 import BeautifulSoup, Tag

# 1. LOADING & METADATA

def load_html(path: Path) -> BeautifulSoup:
    """Read an HTML file and return a parsed DOM tree."""
    with open(path, encoding="utf-8") as f:
        return BeautifulSoup(f, "lxml")


def generate_chunk_id(doc_name: str, article: str, para: str, text: str) -> str:
    """Generate a stable unique ID for each chunk."""
    base = f"{doc_name}|{article}|{para}"
    # Add text hash for uniqueness when multiple chunks have same location
    txt_hash = sha256(text.encode()).hexdigest()[:8]
    return f"{base}|{txt_hash}"


def extract_document_title(soup: BeautifulSoup) -> str:
    """Get the full legislation title from the EUR-Lex header."""
    parts: list[str] = []
    for p in soup.select(
        "div.eli-main-title p.title-doc-first, "
        "div.eli-main-title p.title-doc-last"
    ):
        txt = clean(p.get_text())
        if txt:
            parts.append(txt)
    return " — ".join(parts) if parts else ""


# Map of known abbreviation fragments → canonical short names.
# Extend this dict when adding new legislation types.
_KNOWN_LABELS: dict[str, str] = {
    "ESPR":  "ESPR",
    "REACH": "REACH",
    "ROHS":  "RoHS",
    "WEE":   "WEEE",
    "WEEE":  "WEEE",
    "CAEP":  "CAEP",
}


def infer_document_short_name(path: Path) -> str:
    """Derive a short label from the filename (e.g. 'WEEE', 'RoHS')."""
    stem = path.stem.upper()
    for fragment, label in _KNOWN_LABELS.items():
        if fragment in stem:
            return label
    return stem.split("-")[0].strip()

# 2. TEXT HELPERS

_WS = re.compile(r"\s+")
_ARROW = re.compile(r"[▼►◄▲][A-Z0-9]+\b")  # amendment markers ▼B, ►M2 …


def clean(text: str) -> str:
    """Normalise whitespace, strip EUR-Lex amendment arrows, trim."""
    text = _ARROW.sub("", text)
    return _WS.sub(" ", text).strip()


def tag_text(tag: Tag) -> str:
    """Recursively extract clean text from an HTML element."""
    return clean(tag.get_text())


# 3. DOM NAVIGATION — hierarchy extraction

def find_chapter(article_tag: Tag) -> str:
    """Walk up from an article <div> to find its chapter/title heading.

    EUR-Lex uses ids like  cpt_I  (chapter) and  tis_I  (title).
    Each has child <p class="title-division-1/2"> with the heading text.
    """
    parent = article_tag.parent
    while parent:
        pid = parent.get("id", "")
        if pid and re.match(r"(cpt_|tis_)", pid):
            parts: list[str] = []
            for p in parent.find_all(
                "p", class_=re.compile(r"title-division-[12]"),
                recursive=False,
            ):
                parts.append(tag_text(p))
            if parts:
                return " — ".join(parts)
        parent = parent.parent
    return ""


def get_article_heading(art: Tag) -> tuple[str, str]:
    """Return (article_number, article_subtitle) for an article div.

    Number comes from  <p class="title-article-norm">.
    Subtitle from  <p class="stitle-article-norm"> or <p class="norm">
    inside the  <div class="eli-title"> wrapper.
    """
    num_tag = art.find("p", class_="title-article-norm")
    number = tag_text(num_tag) if num_tag else ""

    subtitle = ""
    title_div = art.find("div", class_="eli-title")
    if title_div:
        st = title_div.find(
            "p", class_=re.compile(r"stitle-article-norm|^norm$")
        )
        if st:
            subtitle = tag_text(st)
    return number, subtitle


# 4. PARAGRAPH & LIST-ITEM EXTRACTION

def extract_paragraphs(article: Tag) -> list[dict]:
    """Extract numbered and unnumbered paragraphs from an article.

    Returns list of dicts:
      { "para_num": "1" | "",  "text": "full paragraph text" }

    List items (a), (b)… stay within their parent paragraph text
    so that "shall/except/where" clauses remain together.
    """
    paragraphs: list[dict] = []
    seen: set[int] = set()

    # ── Numbered paragraphs: <div class="norm"> with <span class="no-parag">
    for div in article.find_all("div", class_="norm", recursive=False):
        span = div.find("span", class_="no-parag")
        if not span:
            continue
        seen.add(id(div))
        para_num = clean(span.get_text()).rstrip(".").strip()

        body_tag = (
            div.find("div", class_="inline-element")
            or div.find("p", class_="inline-element")
        )
        text = tag_text(body_tag) if body_tag else clean(
            div.get_text().replace(span.get_text(), "", 1)
        )
        if text:
            paragraphs.append({"para_num": para_num, "text": text})

    # ── Unnumbered standalone <p class="norm"> (direct children)
    for p in article.find_all("p", class_="norm", recursive=False):
        if id(p) in seen:
            continue
        if p.find_parent("div", class_="eli-title"):
            continue
        txt = tag_text(p)
        if txt and len(txt) > 15:
            paragraphs.append({"para_num": "", "text": txt})

    return paragraphs


# 5. ANNEX EXTRACTION

_MAX_ANNEX_CHARS = 2000  # truncate very large annexes


def extract_annexes(soup: BeautifulSoup, doc_name: str, source_file: str, version: str) -> list[dict]:
    """Extract annexes as coarse-grained chunks (title + body preview)."""
    chunks: list[dict] = []
    for anx in soup.find_all("div", id=re.compile(r"^anx_")):
        title_parts: list[str] = []
        for cls in ("title-annex-1", "title-annex-2"):
            for p in anx.find_all("p", class_=cls, limit=3):
                title_parts.append(tag_text(p))
        title = " — ".join(t for t in title_parts if t)
        if not title:
            continue

        body = tag_text(anx)
        for t in title_parts:
            body = body.replace(t, "", 1).strip()
        if len(body) > _MAX_ANNEX_CHARS:
            body = body[:_MAX_ANNEX_CHARS] + " [...]"

        chunk_id = generate_chunk_id(doc_name, title, "", body)
        chunks.append({
            "id": chunk_id,
            "document": doc_name,
            "source_file": source_file,
            "version": version,
            "chapter": "Annex",
            "article": title,
            "article_subtitle": "",
            "paragraph": "",
            "char_offset": 0,
            "text": body,
        })
    return chunks


# 6. MAIN PARSE PIPELINE

def parse_eurlex_html(path: Path) -> list[dict]:
    """Parse one EUR-Lex HTML file into a list of provision chunks.

    Each chunk is a dict with keys:
      id, document, source_file, version, chapter, article,
      article_subtitle, paragraph, char_offset, text
    """
    soup = load_html(path)
    doc_name = infer_document_short_name(path)
    source_file = path.name
    # Extract version from filename (e.g., "01.01.2025" from "ROHS-01.01.2025.html")
    version = ""
    stem_parts = path.stem.split("-")
    if len(stem_parts) > 1:
        version = "-".join(stem_parts[1:])

    rows: list[dict] = []

    # ── Articles ──
    for art in soup.find_all(
        "div", class_="eli-subdivision", id=re.compile(r"^art_")
    ):
        chapter = find_chapter(art)
        art_num, art_sub = get_article_heading(art)
        paras = extract_paragraphs(art)

        if paras:
            for p in paras:
                text = p["text"]
                chunk_id = generate_chunk_id(doc_name, art_num, p["para_num"], text)
                rows.append({
                    "id": chunk_id,
                    "document": doc_name,
                    "source_file": source_file,
                    "version": version,
                    "chapter": chapter,
                    "article": art_num,
                    "article_subtitle": art_sub,
                    "paragraph": p["para_num"],
                    "char_offset": 0,  # placeholder - full DOM traversal needed for precise offset
                    "text": text,
                })
        else:
            # Fallback: article has no parseable paragraphs
            full = tag_text(art)
            if full and len(full) > 20:
                chunk_id = generate_chunk_id(doc_name, art_num, "", full)
                rows.append({
                    "id": chunk_id,
                    "document": doc_name,
                    "source_file": source_file,
                    "version": version,
                    "chapter": chapter,
                    "article": art_num,
                    "article_subtitle": art_sub,
                    "paragraph": "",
                    "char_offset": 0,
                    "text": full,
                })

    # ── Annexes ──
    rows.extend(extract_annexes(soup, doc_name, source_file, version))

    return rows


# 7. MULTI-FILE PROCESSING & CSV OUTPUT

FIELDNAMES = [
    "id", "document", "source_file", "version",
    "chapter", "article", "article_subtitle",
    "paragraph", "char_offset", "text",
]


def collect_all(input_path: Path) -> list[dict]:
    """Process one file or every .html file in a directory."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.html"))
    if not files:
        raise SystemExit(f"No HTML files found in {input_path}")

    all_rows: list[dict] = []
    for f in files:
        print(f"  Parsing {f.name} …", end=" ")
        rows = parse_eurlex_html(f)
        print(f"{len(rows)} chunks")
        all_rows.extend(rows)
    return all_rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write provision chunks to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def main(input_path: str, output_path: str) -> None:
    rows = collect_all(Path(input_path))
    write_csv(rows, Path(output_path))
    print(f"\nWrote {len(rows)} total chunks → {output_path}")


# 8. CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Chunk EUR-Lex legislation HTML into per-provision CSV rows."
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="Path to a single .html file or a directory of .html files",
    )
    p.add_argument(
        "-o", "--output", required=True,
        help="Output CSV path",
    )
    args = p.parse_args()
    main(args.input, args.output)