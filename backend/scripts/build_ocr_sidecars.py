"""Generate version-checked OCR text for image-only PDFs.

This is an optional maintenance command. Normal users do not need Tesseract because the two
checked sidecars are bundled with the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import fitz

from backend.ir_system.config import PROJECT_ROOT, settings


def build_sidecar(pdf_path: Path, *, dpi: int, language: str, force: bool = False) -> Path:
    pages = []
    with fitz.open(pdf_path) as document:
        for page in document:
            native_text = page.get_text("text")
            if native_text.strip():
                pages.append(native_text)
                continue
            text_page = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
            pages.append(page.get_text("text", textpage=text_page))
    payload = {
        "source_pdf": pdf_path.name,
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "engine": "Tesseract 5 / PyMuPDF",
        "language": language,
        "dpi": dpi,
        "generated_on": date.today().isoformat(),
        "visually_verified": False,
        "pages": pages,
    }
    settings.ocr_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.ocr_dir / f"{pdf_path.stem}.json"
    if output_path.exists() and not force:
        raise FileExistsError(f"Sidecar already exists: {output_path}. Use --force to replace it.")
    temporary_path = output_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OCR sidecars for image-only corpus PDFs.")
    parser.add_argument("files", nargs="*", help="PDF basenames; defaults to every image-only PDF")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--force", action="store_true", help="Replace an existing reviewed sidecar")
    args = parser.parse_args()

    requested = [settings.corpus_dir / file_name for file_name in args.files]
    candidates = requested or sorted(settings.corpus_dir.glob("*.pdf"))
    generated = 0
    for pdf_path in candidates:
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise SystemExit(f"PDF not found: {pdf_path}")
        with fitz.open(pdf_path) as document:
            needs_ocr = any(not page.get_text("text").strip() for page in document)
        if not needs_ocr:
            continue
        output_path = build_sidecar(
            pdf_path, dpi=args.dpi, language=args.language, force=args.force
        )
        print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")
        generated += 1
    print(f"Generated {generated} OCR sidecar(s).")
    if generated:
        print(
            "Review every page against a rendered PDF, correct OCR errors, then set "
            "visually_verified to true. Unreviewed sidecars are not loaded."
        )


if __name__ == "__main__":
    main()
