import re
import json
from bs4 import BeautifulSoup
from pathlib import Path

# --- Patterns for EU legal structure ---
RE_RECITAL_START = re.compile(r"^\(\d+\)\s*$")
RE_RECITAL_NUM = re.compile(r"^\((\d+)\)\s*$")

RE_ARTICLE = re.compile(r"^Article\s+(\d+)\s*$", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^CHAPTER\s+([IVXLC]+)\s*$", re.IGNORECASE)
RE_SECTION = re.compile(r"^SECTION\s+(\d+)\s*$", re.IGNORECASE)
RE_ANNEX = re.compile(r"^ANNEX\s+([IVXLC]+)\s*$", re.IGNORECASE)

RE_PARA_NUM = re.compile(r"^(\d+)\.\s+(.*)$")          # "1. Text..."
RE_POINT = re.compile(r"^\(([a-z])\)\s+(.*)$")         # "(a) Text..."
RE_SUBPOINT = re.compile(r"^\(([ivxlcdm]+)\)\s+(.*)$", re.IGNORECASE)  # "(i) Text..."

def html_to_lines(html_path: Path) -> list[str]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    text = soup.get_text("\n")
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            lines.append(ln)
    return lines

def emit(chunks, doc_meta, section_path, chunk_id, text):
    chunks.append({
        **doc_meta,
        "section_path": section_path[:],
        "chunk_id": chunk_id,
        "text": text.strip(),
    })

def chunk_directive_or_regulation(lines: list[str], doc_meta: dict) -> list[dict]:
    chunks = []

    # State
    in_recitals = False
    in_articles = False
    in_annex = False

    current_chapter = None
    current_section = None
    current_article = None
    current_annex = None

    # Buffers
    recital_num = None
    recital_buf = []

    article_level = {
        "para": None,
        "point": None,
        "subpoint": None,
    }
    article_buf = []

    def flush_recital():
        nonlocal recital_num, recital_buf
        if recital_num is not None and recital_buf:
            cid = f"{doc_meta['doc_short']}:recital:{recital_num}"
            emit(chunks, doc_meta, ["Preamble", f"Recital ({recital_num})"], cid, " ".join(recital_buf))
        recital_num, recital_buf = None, []

    def flush_article_buffer():
        nonlocal article_buf, current_article, article_level
        if current_article and article_buf:
            # build chunk id at finest available granularity
            parts = [f"{doc_meta['doc_short']}:art{current_article}"]
            path = []
            if current_chapter: path.append(f"CHAPTER {current_chapter}")
            if current_section: path.append(f"SECTION {current_section}")
            path.append(f"Article {current_article}")

            if article_level["para"]:
                parts.append(f"para{article_level['para']}")
                path.append(f"Paragraph {article_level['para']}")
            if article_level["point"]:
                parts.append(f"point{article_level['point']}")
                path.append(f"Point ({article_level['point']})")
            if article_level["subpoint"]:
                parts.append(f"sub{article_level['subpoint']}")
                path.append(f"Subpoint ({article_level['subpoint']})")

            cid = ":".join(parts)
            emit(chunks, doc_meta, path, cid, " ".join(article_buf))

        article_buf = []

    for ln in lines:
        # --- Enter recitals ---
        if ln == "Whereas:":
            flush_article_buffer()
            in_recitals, in_articles, in_annex = True, False, False
            continue

        # Typical boundary into Articles (phrasing differs slightly across acts)
        if "HAVE ADOPTED THIS" in ln.upper():
            flush_recital()
            in_recitals = False
            in_articles = True
            continue

        # --- Headings ---
        m = RE_CHAPTER.match(ln)
        if m:
            flush_article_buffer(); flush_recital()
            current_chapter = m.group(1)
            current_section = None
            continue

        m = RE_SECTION.match(ln)
        if m:
            flush_article_buffer(); flush_recital()
            current_section = m.group(1)
            continue

        m = RE_ANNEX.match(ln)
        if m:
            flush_article_buffer(); flush_recital()
            in_annex, in_articles, in_recitals = True, False, False
            current_annex = m.group(1)
            continue

        m = RE_ARTICLE.match(ln)
        if m:
            flush_article_buffer(); flush_recital()
            in_articles, in_recitals, in_annex = True, False, False
            current_article = m.group(1)
            # reset nesting
            article_level = {"para": None, "point": None, "subpoint": None}
            continue

        # --- Recitals parsing ---
        if in_recitals:
            m = RE_RECITAL_NUM.match(ln)
            if m:
                flush_recital()
                recital_num = m.group(1)
            else:
                # Only buffer text once we've seen a recital number
                if recital_num is not None:
                    recital_buf.append(ln)
            continue

        # --- Articles parsing ---
        if in_articles and current_article:
            # Paragraph start
            m = RE_PARA_NUM.match(ln)
            if m:
                flush_article_buffer()
                article_level = {"para": m.group(1), "point": None, "subpoint": None}
                article_buf.append(m.group(2))
                continue

            # Point (a)(b)...
            m = RE_POINT.match(ln)
            if m:
                flush_article_buffer()
                if article_level["para"] is None:
                    article_level["para"] = "0"  # unnumbered paragraph fallback
                article_level["point"] = m.group(1)
                article_level["subpoint"] = None
                article_buf.append(m.group(2))
                continue

            # Subpoint (i)(ii)...
            m = RE_SUBPOINT.match(ln)
            if m and article_level["point"] is not None:
                flush_article_buffer()
                article_level["subpoint"] = m.group(1).lower()
                article_buf.append(m.group(2))
                continue

            # Normal continuation
            article_buf.append(ln)
            continue

        # --- Annex parsing (simple baseline) ---
        if in_annex and current_annex:
            # Treat each non-empty line as annex content; improve later if needed
            cid = f"{doc_meta['doc_short']}:annex:{current_annex}"
            emit(chunks, doc_meta, ["ANNEX " + current_annex], cid, ln)
            continue

    flush_recital()
    flush_article_buffer()
    return chunks

def run(html_file: str, doc_id: str, title: str, doc_type: str, doc_short: str, out_jsonl: str):
    lines = html_to_lines(Path(html_file))
    doc_meta = {
        "doc_id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "doc_short": doc_short,
    }
    chunks = chunk_directive_or_regulation(lines, doc_meta)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks to {out_jsonl}")

# Example:
# run("rohs_2011_65_eu.html", "CELEX:32011L0065", "RoHS (recast)", "Directive", "rohs2011", "rohs2011_chunks.jsonl")
