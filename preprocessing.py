import re
import statistics
from collections import Counter
from pathlib import Path

import pymupdf
from tqdm.auto import tqdm

TERMINAL = (".", "!", "?", "…", ":", "»", "”", '"')
HEADING = re.compile(r"^CAP[IÍ]TULO\b", re.IGNORECASE)
DASHES = ("—", "–")


# ---------- 1. Ler as linhas de cada página ----------

def read_page_lines(page):
    """Devolve as linhas visuais da página, com a linha de base (y) estável.
    O travessão que vem noutro fragmento é juntado ao texto da mesma linha."""
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
                "y": spans[0]["origin"][1],          # linha de base, não o topo da caixa
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


# ---------- 2. Limpar números de página e cabeçalhos ----------

def drop_page_number(lines):
    """Só remove números isolados na primeira ou na última linha da página."""
    if lines and re.fullmatch(r"\d{1,4}", lines[-1]["text"]):
        lines = lines[:-1]
    if lines and re.fullmatch(r"\d{1,4}", lines[0]["text"]):
        lines = lines[1:]
    return lines


def drop_running_headers(pages, min_share=0.3):
    """Remove cabeçalhos/rodapés que se repetem no topo ou fundo de muitas páginas."""
    n = len(pages)
    firsts = Counter(p[0]["text"] for p in pages if p)
    lasts = Counter(p[-1]["text"] for p in pages if p)
    bad = {t for t, c in (firsts + lasts).items()
           if c >= max(3, min_share * n) and len(t) <= 60 and not HEADING.match(t)}
    if bad:
        print("Cabeçalhos/rodapés removidos:", bad)
    cleaned = []
    for p in pages:
        if p and p[0]["text"] in bad:
            p = p[1:]
        if p and p[-1]["text"] in bad:
            p = p[:-1]
        if p:
            cleaned.append(p)
    return cleaned


# ---------- 3. Medir o espaçamento do livro inteiro ----------

def layout_stats(pages):
    gaps = []
    for lines in pages:
        for a, b in zip(lines, lines[1:]):
            g = b["y"] - a["y"]
            if g > 5:
                gaps.append(g)
    line_gap = statistics.median(gaps)                       # espaçamento normal
    bigger = [round(g) for g in gaps if 1.15 * line_gap < g < 2.5 * line_gap]
    para_gap = Counter(bigger).most_common(1)[0][0] if bigger else 1.3 * line_gap
    threshold = (line_gap + para_gap) / 2                    # ponto médio
    rights = sorted(l["x1"] for p in pages for l in p)
    right = rights[int(0.9 * len(rights))]                   # margem direita
    return line_gap, para_gap, threshold, right


# ---------- 4. Decidir onde começa um parágrafo ----------

def starts_new_paragraph(prev, cur, threshold, right):
    em = cur["size"]
    gap = cur["y"] - prev["y"]
    ended = prev["text"].rstrip().endswith(TERMINAL)
    heading = bool(HEADING.match(prev["text"]) or HEADING.match(cur["text"]))

    if gap < 0:   # mudança de página: não há sinal de espaço, usa-se a forma da linha
        dialogue = cur["text"].startswith(DASHES) and ended
        short_end = ended and prev["x1"] < right - 3 * em
        return heading or dialogue or short_end

    return gap > threshold or heading


def join_lines(a, b):
    if a.endswith("-") and b[:1].islower():        # "arquiteto-" + "decorador"
        return a + b
    return a + " " + b


# ---------- 5. Extrair o PDF ----------

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


# ---------- 6. Limpeza final ----------

def normalize_extracted_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [re.sub(r"[ \t]+", " ", p).strip() for p in paragraphs]
    return "\n\n".join(p for p in paragraphs if p)


def extract_novel(text):
    """Mantém o texto a partir do Capítulo I."""
    m = re.search(r"\bCAP[IÍ]TULO\s+I\b", text, flags=re.IGNORECASE)
    if m is None:
        raise ValueError("Chapter I was not found. Check the extracted PDF text.")
    return text[m.start():].strip()


def clean_extracted_text(text, min_repeats=5, max_len=60):
    """Remove parágrafos curtos, em maiúsculas, que se repetem muitas vezes
    (cabeçalhos que tenham escapado). Mostra o que removeu."""
    paras = text.split("\n\n")
    counts = Counter(p.strip() for p in paras)
    remove = {p for p, c in counts.items()
              if c >= min_repeats and len(p) <= max_len
              and not p.startswith(DASHES)
              and not HEADING.match(p)
              and p == p.upper()}
    if remove:
        print("Removidos:", remove)
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