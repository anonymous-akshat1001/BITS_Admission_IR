"""Page-aware PDF loading, cleanup, and structure-preserving chunking."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .models import Chunk, CorpusSummary
from .preprocessing import normalize_unicode


logger = logging.getLogger(__name__)

TITLE_OVERRIDES = {
    "Travel-Grant-for-Registered-Ph.d-Student.pdf": "National Institute Travel Grant for Registered PhD Students",
    "Institute-Fellowship-policy-for_PhD.pdf": "Institute Fellowship Policy for PhD Students",
}

HEADING_RE = re.compile(r"^(?:section\s+)?(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Z\s/&(),.-]{4,}$")
BULLET_RE = re.compile(r"^(?:[•*-]|\(?[a-z0-9ivx]+[.)])\s+", re.I)
PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d+(?:\s+of\s+\d+)?$", re.I)
SENTENCE_END_RE = re.compile(r"[.!?:;]['\"]?$|\)$")


def humanize_file_name(file_name: str) -> str:
    if file_name in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[file_name]
    stem = Path(file_name).stem
    stem = re.sub(r"^\d+[_ -]+", "", stem)
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    replacements = {
        "Ph d": "PhD", "Ph D": "PhD", "phd": "PhD", "Drc": "DRC", "Dac": "DAC",
        "Agsrd": "AGSRD", "Gcir": "GCIR", "Ta Da": "TA/DA", "Sop": "SOP",
    }
    title = stem.title()
    for old, new in replacements.items():
        title = title.replace(old, new)
    return title


def _normalized_line_key(line: str) -> str:
    key = re.sub(r"\d+", "#", normalize_unicode(line).lower())
    return re.sub(r"\W+", " ", key).strip()


def detect_repeated_marginal_lines(page_lines: Sequence[Sequence[str]]) -> Set[str]:
    """Detect headers/footers repeated on many pages of a multi-page document."""
    if len(page_lines) < 4:
        return set()
    counts: Counter[str] = Counter()
    for lines in page_lines:
        marginal = list(lines[:3]) + list(lines[-3:])
        counts.update({_normalized_line_key(line) for line in marginal if len(line.strip()) >= 4})
    threshold = max(3, round(len(page_lines) * 0.45))
    return {key for key, count in counts.items() if key and count >= threshold}


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not 3 <= len(stripped) <= 120:
        return False
    alpha = [character for character in stripped if character.isalpha()]
    uppercase_ratio = sum(character.isupper() for character in alpha) / max(1, len(alpha))
    numbered_heading = bool(re.match(r"^\d+(?:\.\d+)+\s+\D", stripped))
    return bool(HEADING_RE.match(stripped)) or uppercase_ratio > 0.82 or numbered_heading


def clean_page_text(raw_text: str, repeated_lines: Set[str] | None = None) -> str:
    """Clean extraction noise while retaining headings, lists, and paragraph boundaries."""
    text = normalize_unicode(raw_text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    raw_lines = text.splitlines()
    repeated_lines = repeated_lines or set()
    lines: List[str] = []
    for raw_line in raw_lines:
        line = re.sub(r"_{3,}", " ", raw_line)
        line = re.sub(r"[\t ]+", " ", line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _normalized_line_key(line) in repeated_lines or PAGE_NUMBER_RE.match(line):
            continue
        lines.append(line)

    paragraphs: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current).strip())
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if _is_heading(line) or BULLET_RE.match(line):
            flush()
            paragraphs.append(line)
            continue
        current.append(line)
        if SENTENCE_END_RE.search(line) or sum(len(part) for part in current) > 450:
            flush()
    flush()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def _split_long_segment(segment: str, target_size: int) -> List[str]:
    if len(segment) <= target_size:
        return [segment]
    sentences = re.split(r"(?<=[.!?])\s+", segment)
    if len(sentences) == 1:
        words = segment.split()
        pieces: List[str] = []
        current: List[str] = []
        for word in words:
            if current and len(" ".join(current + [word])) > target_size:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return pieces
    pieces = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target_size:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_page(text: str, target_size: int = 900, overlap: int = 140) -> List[Tuple[str, str]]:
    """Pack paragraph/list segments without crossing page boundaries."""
    raw_segments = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
    segments: List[str] = []
    for segment in raw_segments:
        segments.extend(_split_long_segment(segment, target_size))

    chunks: List[Tuple[str, str]] = []
    current: List[str] = []
    current_length = 0
    active_heading = ""
    chunk_heading = ""

    def flush() -> None:
        nonlocal current, current_length, chunk_heading
        if not current:
            return
        chunks.append(("\n\n".join(current).strip(), chunk_heading))
        overlap_segments: List[str] = []
        overlap_length = 0
        for segment in reversed(current):
            if overlap_segments and overlap_length + len(segment) > overlap:
                break
            overlap_segments.insert(0, segment)
            overlap_length += len(segment)
            if overlap_length >= overlap:
                break
        current = overlap_segments
        current_length = sum(len(segment) + 2 for segment in current)
        chunk_heading = active_heading

    for segment in segments:
        if _is_heading(segment):
            active_heading = segment
        if current and current_length + len(segment) + 2 > target_size:
            flush()
        if not current:
            chunk_heading = active_heading
        if not current or current[-1] != segment:
            current.append(segment)
            current_length += len(segment) + 2
    flush()
    return chunks


def _extract_page_text(page, *, enable_ocr: bool, ocr_language: str) -> Tuple[str, bool]:
    text = page.get_text("text")
    if text.strip() or not enable_ocr:
        return text, False
    try:
        text_page = page.get_textpage_ocr(language=ocr_language, dpi=180, full=True)
        return page.get_text("text", textpage=text_page), True
    except Exception as exc:  # OCR requires an optional system Tesseract installation.
        logger.warning("OCR failed for page %s: %s", page.number + 1, exc)
        return "", True


def extract_structured_tables(page, page_text: str) -> str:
    """Append row-wise text for PDF tables whose visual column order is otherwise lost."""
    lowered = page_text.lower()
    likely_table = (
        bool(re.search(r"\b(?:table|sr\.?\s*no|s\.?\s*no|distribution|budget head)\b", lowered))
        or page_text.count("%") >= 2
        or lowered.count("amount") >= 3
    )
    if not likely_table:
        return ""
    try:
        tables = page.find_tables().tables
    except Exception as exc:
        logger.debug("Table detection failed on page %s: %s", page.number + 1, exc)
        return ""
    if not tables:
        return ""

    flat_page = re.sub(r"\s+", " ", page_text)
    condition_a_match = re.search(
        r"\*\s*(For Projects with\s*[^.]+?)(?:as per\s+column A|\.)", flat_page, re.I
    )
    condition_b_match = re.search(
        r"#\s*(For Projects with\s*[^,.:]+)", flat_page, re.I
    )
    def clean_condition(value: str) -> str:
        cleaned = re.sub(r"^for\s+", "", value.strip(" :"), flags=re.I)
        cleaned = re.sub(r"≤\s*10%", "10% or less", cleaned)
        cleaned = re.sub(r">\s*10%", "more than 10%", cleaned)
        return cleaned

    conditions = {
        "A*": clean_condition(condition_a_match.group(1)) if condition_a_match else "column A",
        "B#": clean_condition(condition_b_match.group(1)) if condition_b_match else "column B",
    }

    rendered_tables: List[str] = []
    for table_number, table in enumerate(tables, start=1):
        rows = [
            [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
            for row in table.extract()
        ]
        if not rows:
            continue
        output_rows = [f"Structured table {table_number}:"]
        has_symbol_columns = len(rows) >= 3 and any(cell in conditions for cell in rows[1])
        if has_symbol_columns:
            subheaders = rows[1]
            for row in rows[2:]:
                labels = [cell for cell in row[:3] if cell]
                if not labels:
                    continue
                values = []
                for column_index in range(3, min(len(row), len(subheaders))):
                    if row[column_index]:
                        symbol = subheaders[column_index]
                        condition = conditions.get(symbol, symbol or f"column {column_index + 1}")
                        values.append(f"{row[column_index]} for {condition.lower()}")
                output_rows.append(f"{' - '.join(labels)}: {'; '.join(values)}.")
        else:
            output_rows.extend(" | ".join(cell for cell in row if cell) for row in rows)
        rendered_tables.append("\n".join(output_rows))
    return "\n\n".join(rendered_tables)


def load_ocr_sidecar(
    pdf_path: Path,
    ocr_dir: Path | None,
    *,
    expected_page_count: int | None = None,
) -> List[str] | None:
    """Load manually reviewed OCR text generated from the exact current PDF."""
    if ocr_dir is None:
        return None
    sidecar_path = ocr_dir / f"{pdf_path.stem}.json"
    if not sidecar_path.is_file():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        pages = payload.get("pages")
        if (
            payload.get("pdf_sha256") != expected_hash
            or payload.get("visually_verified") is not True
            or not isinstance(pages, list)
            or (expected_page_count is not None and len(pages) != expected_page_count)
        ):
            logger.warning("Ignoring stale or invalid OCR sidecar: %s", sidecar_path)
            return None
        return [str(page) for page in pages]
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not read OCR sidecar %s: %s", sidecar_path, exc)
        return None


def load_corpus(
    corpus_dir: Path,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 140,
    enable_ocr: bool = False,
    ocr_language: str = "eng",
    ocr_dir: Path | None = None,
) -> Tuple[List[Chunk], CorpusSummary]:
    """Extract every PDF into page-aware chunks and a diagnostic summary."""
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - exercised only in a broken environment.
        raise RuntimeError("PyMuPDF is required. Install backend/requirements.txt first.") from exc

    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    chunks: List[Chunk] = []
    summary = CorpusSummary()
    for pdf_path in sorted(corpus_dir.glob("*.pdf"), key=lambda path: path.name.lower()):
        summary.document_count += 1
        document_id = hashlib.sha1(pdf_path.name.encode("utf-8")).hexdigest()[:12]
        title = humanize_file_name(pdf_path.name)
        with fitz.open(pdf_path) as document:
            sidecar_pages = load_ocr_sidecar(
                pdf_path, ocr_dir, expected_page_count=len(document)
            )
            summary.page_count += len(document)
            page_texts: List[str] = []
            ocr_attempted = False
            for page_index, page in enumerate(document):
                text = page.get_text("text")
                attempted = False
                if not text.strip() and sidecar_pages and page_index < len(sidecar_pages):
                    text = sidecar_pages[page_index]
                elif not text.strip():
                    text, attempted = _extract_page_text(
                        page, enable_ocr=enable_ocr, ocr_language=ocr_language
                    )
                if text.strip():
                    structured_tables = extract_structured_tables(page, text)
                    if structured_tables:
                        text = f"{text.rstrip()}\n\n{structured_tables}"
                ocr_attempted = ocr_attempted or attempted
                page_texts.append(text)

        if sidecar_pages and any(text.strip() for text in page_texts):
            summary.ocr_sidecar_documents.append(pdf_path.name)

        page_lines = [[line.strip() for line in text.splitlines() if line.strip()] for text in page_texts]
        repeated_lines = detect_repeated_marginal_lines(page_lines)
        document_chunk_count = 0
        for page_number, raw_text in enumerate(page_texts, start=1):
            cleaned = clean_page_text(raw_text, repeated_lines)
            if not cleaned:
                continue
            for page_chunk_index, (text, section) in enumerate(
                chunk_page(cleaned, target_size=chunk_size, overlap=chunk_overlap), start=1
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document_id}-p{page_number}-c{page_chunk_index}",
                        document_id=document_id,
                        file_name=pdf_path.name,
                        title=title,
                        page_start=page_number,
                        page_end=page_number,
                        text=text,
                        section=section,
                    )
                )
                document_chunk_count += 1

        if document_chunk_count:
            summary.searchable_document_count += 1
        else:
            summary.scanned_or_empty_documents.append(pdf_path.name)
            if ocr_attempted:
                summary.ocr_failures.append(pdf_path.name)
            # A title-only record makes the correct scanned PDF discoverable without pretending
            # that its contents were extracted.
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-metadata",
                    document_id=document_id,
                    file_name=pdf_path.name,
                    title=title,
                    page_start=1,
                    page_end=max(1, len(page_texts)),
                    text="",
                    section="Scanned PDF - text extraction unavailable",
                    text_available=False,
                )
            )
    summary.chunk_count = len(chunks)
    return chunks, summary
