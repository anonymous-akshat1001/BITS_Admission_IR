import hashlib
import json

from backend.ir_system.corpus import (
    chunk_page,
    clean_page_text,
    humanize_file_name,
    load_ocr_sidecar,
)
from backend.ir_system.preprocessing import explicit_identifier_terms, query_terms, tokenize


def test_cleaning_preserves_unicode_and_structure() -> None:
    raw = "SECTION ONE\n\nThe maxi-\nmum grant is ₹25,000.\n\n• Submit bills."
    cleaned = clean_page_text(raw)

    assert "maximum" in cleaned
    assert "₹25,000" in cleaned
    assert "SECTION ONE" in cleaned
    assert "• Submit bills." in cleaned


def test_query_normalization_and_expansion() -> None:
    original, weighted = query_terms("What is the Ph.D. stipend for a full-time scholar?")

    assert "phd" in original
    assert "fulltime" in original
    assert weighted["fellowship"] < weighted["stipend"]


def test_page_chunking_respects_target_and_retains_heading() -> None:
    text = "ELIGIBILITY\n\n" + "First requirement. " * 35 + "\n\nSecond requirement. " * 35
    chunks = chunk_page(text, target_size=260, overlap=50)

    assert len(chunks) > 2
    assert all(chunk.strip() for chunk, _ in chunks)
    assert any(section == "ELIGIBILITY" for _, section in chunks)


def test_humanized_scanned_document_title() -> None:
    assert humanize_file_name("Travel-Grant-for-Registered-Ph.d-Student.pdf").startswith(
        "National Institute Travel Grant"
    )


def test_tokenizer_keeps_contrast_terms() -> None:
    tokens = tokenize("less than 10% versus more than 10%")
    assert "less" in tokens
    assert "more" in tokens
    assert "10" in tokens


def test_stopword_is_removed_before_stemming() -> None:
    assert "someth" not in tokenize("Can you say something about this policy?")


def test_slash_separates_policy_acronyms_and_alternatives() -> None:
    tokens = tokenize("TA/DA for international/national travel")

    assert {"ta", "da", "international", "national", "travel"} <= set(tokens)


def test_ta_da_terms_are_domain_anchors_with_explainable_expansion() -> None:
    original, weighted = query_terms("TA/DA reimbursement form")

    assert {"ta", "da", "reimbursement"} <= set(original)
    assert weighted["travel"] < weighted["ta"]


def test_ta_da_notation_and_separate_identifiers_are_normalized() -> None:
    for notation in (
        "T.A./D.A.", "TA/DA", "TA-DA", "T. A. / D. A.", "T.A. & D.A.",
        "T.A. and D.A.", "TA.DA", "T.A.D.A.",
    ):
        assert tokenize(notation) == ["ta", "da"]

    assert explicit_identifier_terms("UGC NET") == ["ugc", "net"]
    assert explicit_identifier_terms("T.A.D.A.") == []


def test_ocr_sidecar_requires_matching_checksum_and_visual_review(tmp_path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"test-pdf-content")
    payload = {
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "visually_verified": False,
        "pages": ["review me"],
    }
    sidecar_path = tmp_path / "scan.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_ocr_sidecar(pdf_path, tmp_path) is None

    payload["visually_verified"] = True
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_ocr_sidecar(pdf_path, tmp_path) == ["review me"]
    assert load_ocr_sidecar(pdf_path, tmp_path, expected_page_count=2) is None
