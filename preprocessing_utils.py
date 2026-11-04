
import re
import pymupdf

from pathlib import Path
from tqdm.auto import tqdm


def normalize_block(text):
    """
    Join visual lines within a PDF text block.
    Preserve punctuation, capitalisation and hyphens.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join all visual lines within this block
    text = re.sub(r"\s*\n\s*", " ", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF and reconstruct paragraphs using
    the vertical spacing between printed lines.

    Normal line spacing and paragraph spacing are distinguished
    using the layout of each page.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    all_paragraphs = []

    for page in tqdm(document, desc="Extracting PDF"):

        page_data = page.get_text("dict", sort=True)
        raw_lines = []

        # Extract text lines and their coordinates
        for block in page_data["blocks"]:

            if block["type"] != 0:
                continue

            for line in block["lines"]:

                text = "".join(
                    span["text"] for span in line["spans"]
                ).strip()

                if not text:
                    continue

                raw_lines.append({
                    "y": line["bbox"][1],
                    "x": line["bbox"][0],
                    "text": text
                })

        if not raw_lines:
            continue

        # Sort by vertical and horizontal position
        raw_lines.sort(key=lambda item: (item["y"], item["x"]))

        # Merge text fragments that belong to the same printed line.
        # This handles cases such as a dash being extracted separately.
        lines = []

        for item in raw_lines:

            if lines and abs(item["y"] - lines[-1]["y"]) < 2:
                lines[-1]["text"] += " " + item["text"]
            else:
                lines.append(item.copy())

        # Remove standalone page numbers
        lines = [
            line for line in lines
            if not re.fullmatch(r"\d{1,4}", line["text"].strip())
        ]

        if not lines:
            continue

        # Calculate typical vertical line spacing
        gaps = [
            lines[i + 1]["y"] - lines[i]["y"]
            for i in range(len(lines) - 1)
            if lines[i + 1]["y"] > lines[i]["y"]
        ]

        normal_spacing = (
            sorted(gaps)[len(gaps) // 2]
            if gaps else 31.5
        )

        # A paragraph gap is larger than normal line spacing.
        # For this PDF, normal spacing is ~31.5 and paragraph
        # spacing is ~41.5 points.
        paragraph_threshold = normal_spacing * 1.2

        page_paragraphs = []
        current_paragraph = lines[0]["text"]
        previous_y = lines[0]["y"]

        for line in lines[1:]:

            vertical_gap = line["y"] - previous_y

            if vertical_gap > paragraph_threshold:

                page_paragraphs.append(current_paragraph)
                current_paragraph = line["text"]

            else:
                current_paragraph += " " + line["text"]

            previous_y = line["y"]

        page_paragraphs.append(current_paragraph)

        # Remove empty paragraphs
        page_paragraphs = [
            p.strip() for p in page_paragraphs if p.strip()
        ]

        if not page_paragraphs:
            continue

        # Reconnect text across page boundaries.
        # A chapter heading or a dialogue beginning with a dash
        # is treated as a new paragraph.
        if all_paragraphs:

            first_text = page_paragraphs[0]

            is_chapter_heading = bool(
                re.match(r"^CAP[IÍ]TULO\b", first_text, re.IGNORECASE)
            )

            is_dialogue = first_text.startswith(("—", "–"))

            if not is_chapter_heading and not is_dialogue:
                all_paragraphs[-1] += " " + first_text
            else:
                all_paragraphs.append(first_text)

            all_paragraphs.extend(page_paragraphs[1:])

        else:
            all_paragraphs.extend(page_paragraphs)

    document.close()

    return "\n\n".join(all_paragraphs)


def normalize_extracted_text(text):
    """
    Normalize whitespace without changing paragraph boundaries.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs = re.split(r"\n\s*\n+", text)

    paragraphs = [
        re.sub(r"[ \t]+", " ", p).strip()
        for p in paragraphs
    ]

    paragraphs = [
        p for p in paragraphs if p
    ]

    return "\n\n".join(paragraphs)


def extract_novel(text):
    """
    Retain the text starting at Chapter I.
    """

    chapter_one = re.search(
        r"\bCAP[IÍ]TULO\s+I\b",
        text,
        flags=re.IGNORECASE
    )

    if chapter_one is None:
        raise ValueError(
            "Chapter I was not found. Check the extracted PDF text."
        )

    return text[chapter_one.start():].strip()


def preprocess_novel(pdf_path, output_path):
    """
    Complete preprocessing pipeline:
    1. Extract text blocks from the PDF.
    2. Normalize whitespace.
    3. Identify Chapter I.
    4. Save the novel.
    """

    full_text = extract_pdf_text(pdf_path)

    normalized_text = normalize_extracted_text(full_text)

    novel_text = extract_novel(normalized_text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        novel_text,
        encoding="utf-8"
    )

    print("\nPreprocessing completed.")
    print("Raw characters:", len(full_text))
    print("Novel characters:", len(novel_text))
    print("Novel words:", len(novel_text.split()))
    print("Saved to:", output_path)

    return novel_text
