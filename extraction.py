import argparse
import os
import re

import fitz  # PyMuPDF


def convert_pdf_to_markdown(pdf_path, output_md_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Input file not found: {pdf_path}")

    markdown_lines = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")

            for block in blocks:
                if block[6] != 0:  # skip image blocks
                    continue

                text = block[4].strip()
                if not text:
                    continue

                if text.isdigit() and len(text) < 4:
                    continue

                if re.match(r'^(CHAPTER \d+|ABSTRACT|DEDICATION|ACKNOWLEDGEMENTS|CURRICULUM VITA)', text, re.IGNORECASE):
                    text = re.sub(r'\n', ' ', text).strip()
                    text = f"\n## {text}\n"

                elif re.match(r'^\d+(\.\d+)+\.?\s+\S', text):
                    text = re.sub(r'\n', ' ', text).strip()
                    text = f"\n### {text}\n"

                else:
                    text = re.sub(r'-\n', '', text)
                    text = re.sub(r'\n', ' ', text)
                    text = f"{text}\n\n"

                markdown_lines.append(text)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.writelines(markdown_lines)

    print(f"Conversion complete. Artifact saved to: {output_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown.")
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument("output", help="Path for the output Markdown file")
    args = parser.parse_args()

    convert_pdf_to_markdown(args.input, args.output)
