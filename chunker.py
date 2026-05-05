from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone


MAX_TOKENS = 768          # target ceiling; sits within the 512-1024 spec range
OVERLAP_RATIO = 0.12      # 12% sliding-window overlap on oversized sections
_CHARS_PER_TOKEN = 4      # rough approximation


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_oversized(section: str, overlap_chars: int) -> list[str]:
    """
    Split a section that exceeds MAX_TOKENS at paragraph boundaries.
    Each new chunk is seeded with the tail of the previous one for overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", section) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if _estimate_tokens(candidate) > MAX_TOKENS and current:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars < len(current) else current
            current = f"{tail}\n\n{para}".strip()
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def chunk_markdown(md_path: str, source_file: str | None = None) -> list[dict]:
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    source_file = source_file or os.path.basename(md_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    overlap_chars = int(MAX_TOKENS * _CHARS_PER_TOKEN * OVERLAP_RATIO)

    # Split at heading boundaries; keep the heading attached to its body
    sections = re.split(r"(?=^#{1,3} )", content, flags=re.MULTILINE)

    chunks: list[dict] = []
    chunk_index = 0

    for section in sections:
        # Strip page-marker comments injected by extraction.py
        clean = re.sub(r"<!--\s*page \d+\s*-->", "", section).strip()
        if not clean:
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)", clean)
        header_title = heading_match.group(2).strip() if heading_match else ""

        sub_chunks = (
            [clean]
            if _estimate_tokens(clean) <= MAX_TOKENS
            else _split_oversized(clean, overlap_chars)
        )

        for text in sub_chunks:
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "text": text,
                    "source_file": source_file,
                    "header_title": header_title,
                    "timestamp": timestamp,
                }
            )
            chunk_index += 1

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a Markdown artifact into RAG-ready JSONL."
    )
    parser.add_argument("input", help="Path to the input .md file")
    parser.add_argument("output", help="Path for the output .jsonl file")
    parser.add_argument(
        "--source", help="Override the source_file label in chunk metadata", default=None
    )
    args = parser.parse_args()

    chunks = chunk_markdown(args.input, source_file=args.source)

    with open(args.output, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Chunking complete. {len(chunks)} chunks written to: {args.output}")


if __name__ == "__main__":
    main()
