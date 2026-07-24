"""Validate PDF extraction and print a reproducible corpus summary."""

from __future__ import annotations

import argparse
import json

from backend.ir_system.config import settings
from backend.ir_system.corpus import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate the bundled PDF corpus.")
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Attempt OCR for image-only pages (requires a local Tesseract installation).",
    )
    args = parser.parse_args()
    chunks, summary = load_corpus(
        settings.corpus_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        enable_ocr=args.ocr,
        ocr_language=settings.ocr_language,
        ocr_dir=settings.ocr_dir,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    print(f"Searchable text characters: {sum(len(chunk.text) for chunk in chunks):,}")
    if summary.scanned_or_empty_documents:
        print("\nDocuments needing OCR:")
        for file_name in summary.scanned_or_empty_documents:
            print(f"- {file_name}")


if __name__ == "__main__":
    main()
