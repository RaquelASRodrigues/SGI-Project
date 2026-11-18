"""
01_preprocess.py
----------------
Reads  Data/Os Maias - Eca de Queiros.pdf  and produces:
  Data/novel_text.txt     clean extracted novel text (starting from Chapter I)
  Data/paragraphs.csv     structured paragraph table with chapter, dialogue flag, and word count

What it does, in order:
  1. Extracts visual text lines and baseline coordinates from each PDF page.
  2. Removes isolated page numbers and repeated running headers/footers.
  3. Measures line vs. paragraph vertical spacing across the book to reconstruct paragraphs.
  4. Strips front matter before Chapter I and splits the text into indexed paragraphs.

Usage:
  pip install pymupdf pandas tqdm
  python 01_preprocess.py
  python 01_preprocess.py --force-pdf
"""
import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd
import pymupdf
from tqdm.auto import tqdm

TERMINAL = (".", "!", "?", "…", ":", "»", "”", '"')
HEADING = re.compile(r"^CAP[IÍ]TULO\b", re.IGNORECASE)
DASHES = ("—", "–")


# ---------- 1. Read lines from each page ----------

def read_page_lines(page):
    """Return visual lines of the page using a stable baseline (y).
    Leading dashes in separate fragments are merged into the same line."""
    data = page.get_text("dict", sort=True)
    frags = []
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            frags.append({
                "y": spans[0]["origin"][1],          # text baseline, not bounding-box top
                "x0": line["bbox"][0],
                "x1": line["bbox"][2],
                "size": max(s["size"] for s in spans),
                "text": "".join(s["text"] for s in line["spans"]).strip(),
            })
    frags.sort(key=lambda f: f["y"])

    groups = []
    for f in frags:
        if groups and abs(f["y"] - groups[-1][0]["y"]) < 0.5 * f["size"]:
            groups[-1].append(f)
        else:
            groups.append([f])

    lines = []
    for g in groups:
        g = sorted(g, key=lambda f: f["x0"])
        lines.append({
            "y": g[0]["y"],
            "x0": g[0]["x0"],
            "x1": max(f["x1"] for f in g),
            "size": max(f["size"] for f in g),
            "text": " ".join(f["text"] for f in g),
        })
    return lines


# ---------- 2. Clean page numbers and running headers ----------

def drop_page_number(lines):
    """Remove standalone numbers on the first or last line of a page."""
    if lines and re.fullmatch(r"\d{1,4}", lines[-1]["text"]):
        lines = lines[:-1]
    if lines and re.fullmatch(r"\d{1,4}", lines[0]["text"]):
        lines = lines[1:]
    return lines


def drop_running_headers(pages, min_share=0.3):
    """Remove headers/footers that repeat at the top or bottom of many pages."""
    n = len(pages)
    firsts = Counter(p[0]["text"] for p in pages if p)
    lasts = Counter(p[-1]["text"] for p in pages if p)
    bad = {t for t, c in (firsts + lasts).items()
           if c >= max(3, min_share * n) and len(t) <= 60 and not HEADING.match(t)}
    if bad:
        print("Removed headers/footers:", bad)
    cleaned = []
    for p in pages:
        if p and p[0]["text"] in bad:
            p = p[1:]
        if p and p[-1]["text"] in bad:
            p = p[:-1]
        if p:
            cleaned.append(p)
    return cleaned


# ---------- 3. Measure spacing across the entire book ----------

def layout_stats(pages):
    gaps = []
    for lines in pages:
        for a, b in zip(lines, lines[1:]):
            g = b["y"] - a["y"]
            if g > 5:
                gaps.append(g)
    line_gap = statistics.median(gaps)                       # standard line spacing
    bigger = [round(g) for g in gaps if 1.15 * line_gap < g < 2.5 * line_gap]
    para_gap = Counter(bigger).most_common(1)[0][0] if bigger else 1.3 * line_gap
    threshold = (line_gap + para_gap) / 2                    # midpoint threshold
    rights = sorted(l["x1"] for p in pages for l in p)
    right = rights[int(0.9 * len(rights))]                   # right margin estimate
    return line_gap, para_gap, threshold, right


# ---------- 4. Decide where a new paragraph starts ----------

def starts_new_paragraph(prev, cur, threshold, right):
    em = cur["size"]
    gap = cur["y"] - prev["y"]
    ended = prev["text"].rstrip().endswith(TERMINAL)
    heading = bool(HEADING.match(prev["text"]) or HEADING.match(cur["text"]))

    if gap < 0:   # page break: no vertical gap signal, rely on line geometry
        dialogue = cur["text"].startswith(DASHES) and ended
        short_end = ended and prev["x1"] < right - 3 * em
        return heading or dialogue or short_end

    return gap > threshold or heading


def join_lines(a, b):
    if a.endswith("-") and b[:1].islower():        # de-hyphenate line wraps ("arquiteto-" + "decorador")
        return a + b
    return a + " " + b


# ---------- 5. Extract text from PDF ----------

def extract_pdf_text(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as doc:
        pages = [drop_page_number(read_page_lines(p))
                 for p in tqdm(doc, desc="Reading PDF")]
    pages = [p for p in pages if p]
    pages = drop_running_headers(pages)

    line_gap, para_gap, threshold, right = layout_stats(pages)
    print(f"line_gap={line_gap:.1f}  para_gap={para_gap}  threshold={threshold:.1f}")

    all_lines = [l for p in pages for l in p]
    paragraphs = [all_lines[0]["text"]]
    for prev, cur in zip(all_lines, all_lines[1:]):
        if starts_new_paragraph(prev, cur, threshold, right):
            paragraphs.append(cur["text"])
        else:
            paragraphs[-1] = join_lines(paragraphs[-1], cur["text"])

    return "\n\n".join(re.sub(r"[ \t]+", " ", p).strip() for p in paragraphs)


# ---------- 6. Final cleanup and paragraph table ----------

def normalize_extracted_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def extract_novel(text):
    """Keep text starting from Chapter I."""
    m = re.search(r"\bCAP[IÍ]TULO\s+I\b", text, flags=re.IGNORECASE)
    if m is None:
        raise ValueError("Chapter I was not found. Check the extracted PDF text.")
    return text[m.start():].strip()


def clean_extracted_text(text, min_repeats=5, max_len=60):
    """Remove short, uppercase paragraphs repeated many times (stray headers)."""
    paras = text.split("\n\n")
    counts = Counter(p.strip() for p in paras)
    remove = {p for p, c in counts.items()
              if c >= min_repeats and len(p) <= max_len
              and not p.startswith(DASHES)
              and not HEADING.match(p)
              and p == p.upper()}
    if remove:
        print("Removed:", remove)
    return "\n\n".join(p for p in paras if p.strip() not in remove)


def preprocess_novel(pdf_path, output_path):
    full_text = extract_pdf_text(pdf_path)
    normalized = normalize_extracted_text(full_text)
    novel_text = extract_novel(normalized)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(novel_text, encoding="utf-8")

    print("\nPreprocessing completed.")
    print("Raw characters:", len(full_text))
    print("Novel characters:", len(novel_text))
    print("Novel words:", len(novel_text.split()))
    print("Saved to:", output_path)
    return novel_text


def build_paragraph_table(text):
    rows, chapter, para_in_ch = [], 0, 0
    for p in text.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        if re.match(r"^CAP[IÍ]TULO\b", p, re.IGNORECASE):
            chapter += 1
            para_in_ch = 0
            continue
        para_in_ch += 1
        rows.append({
            "paragraph_id": len(rows),
            "chapter": chapter,
            "para_in_chapter": para_in_ch,
            "is_dialogue": p.startswith(DASHES),
            "n_words": len(p.split()),
            "text": p,
        })
    return pd.DataFrame(rows)


# ---------- 7. Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--pdf", default="Os Maias - Eca de Queiros.pdf")
    ap.add_argument("--force-pdf", action="store_true",
                    help="Re-extract the PDF even if novel_text.txt already exists")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pdf_path = data_dir / args.pdf
    novel_text_path = data_dir / "novel_text.txt"

    if args.force_pdf or not novel_text_path.exists():
        novel_text = preprocess_novel(pdf_path, novel_text_path)
    else:
        novel_text = novel_text_path.read_text(encoding="utf-8")

    cleaned_text = clean_extracted_text(novel_text)
    df = build_paragraph_table(cleaned_text)
    out_csv = data_dir / "paragraphs.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Saved {len(df)} paragraphs ({df['chapter'].nunique()} chapters) to {out_csv}")


if __name__ == "__main__":
    main()