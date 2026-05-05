from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import fitz  # PyMuPDF


def _collect_font_sizes(doc: fitz.Document) -> Counter:
    """First pass: accumulate font sizes weighted by character count."""
    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    t = span["text"].strip()
                    if t:
                        sizes[round(span["size"], 1)] += len(t)
    return sizes


def _heading_thresholds(sizes: Counter) -> tuple[list[tuple[float, int]], float]:
    """
    Derive H1/H2/H3 font-size cutoffs from the document-wide size distribution.

    The most-used size is treated as body text. Any size more than 1pt larger
    is a candidate heading; the top three candidate sizes map to H1/H2/H3.

    Returns ([(min_size, level), ...] sorted descending, body_size).
    """
    if not sizes:
        return [], 11.0

    body_size: float = sizes.most_common(1)[0][0]
    candidates = sorted(
        [s for s in sizes if s > body_size + 1.0], reverse=True
    )

    levels = [(size, idx + 1) for idx, size in enumerate(candidates[:3])]
    return levels, body_size


def _dominant_size(block: dict) -> float:
    """Return the largest font size found in a text block."""
    max_size = 0.0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span["text"].strip():
                max_size = max(max_size, span["size"])
    return round(max_size, 1)


def _block_text(block: dict) -> str:
    """Reconstruct raw text from a dict-mode block."""
    return "\n".join(
        "".join(span["text"] for span in line["spans"])
        for line in block.get("lines", [])
    )


_KNOWN_HEADING = re.compile(
    r"^(CHAPTER \d+|ABSTRACT|DEDICATION|ACKNOWLEDGEMENTS|CURRICULUM VITA)",
    re.IGNORECASE,
)
_NUMBERED_SECTION = re.compile(r"^\d+(\.\d+)+\.?\s+\S")
_LEVEL_PREFIX = {1: "#", 2: "##", 3: "###"}


def convert_pdf_to_markdown(pdf_path: str, output_md_path: str) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input file not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        sizes = _collect_font_sizes(doc)
        heading_levels, body_size = _heading_thresholds(sizes)

        markdown_lines: list[str] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            markdown_lines.append(f"\n<!-- page {page_num + 1} -->\n")

            for block in page.get_text("dict", sort=True)["blocks"]:
                if block["type"] != 0:
                    continue

                raw = _block_text(block).strip()
                if not raw:
                    continue

                # Drop lone page numbers
                if raw.isdigit() and len(raw) < 4:
                    continue

                dom_size = _dominant_size(block)

                # --- Heading classification ---
                heading_level: int | None = None

                # 1. Font-size signal (primary)
                if dom_size > body_size + 1.0:
                    for threshold, level in heading_levels:
                        if dom_size >= threshold - 0.5:
                            heading_level = level
                            break

                # 2. Keyword fallback (e.g. "ABSTRACT" in body-size font)
                if heading_level is None and _KNOWN_HEADING.match(raw):
                    heading_level = 2

                # 3. Numbered-section fallback (e.g. "2.3 Methodology")
                if heading_level is None and _NUMBERED_SECTION.match(raw):
                    heading_level = 3

                # --- Emit ---
                if heading_level is not None:
                    clean = re.sub(r"\s+", " ", raw).strip()
                    prefix = _LEVEL_PREFIX.get(heading_level, "###")
                    markdown_lines.append(f"\n{prefix} {clean}\n")
                else:
                    text = re.sub(r"-\n", "", raw)
                    text = re.sub(r"\n", " ", text)
                    markdown_lines.append(f"{text}\n\n")

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.writelines(markdown_lines)

    print(f"Conversion complete. Artifact saved to: {output_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown.")
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument("output", help="Path for the output Markdown file")
    args = parser.parse_args()

    convert_pdf_to_markdown(args.input, args.output)
