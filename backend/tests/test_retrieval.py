from backend.ir_system.answering import extractive_answer
from backend.ir_system.models import Chunk
from backend.ir_system.retrieval import HybridIndex


def make_chunk(identifier: str, title: str, text: str, *, available: bool = True) -> Chunk:
    return Chunk(
        chunk_id=identifier,
        document_id=identifier,
        file_name=f"{identifier}.pdf",
        title=title,
        page_start=1,
        page_end=1,
        text=text,
        text_available=available,
    )


def test_contrast_term_prevents_international_national_confusion() -> None:
    chunks = [
        make_chunk(
            "national",
            "National Institute Travel Grant for Registered PhD Students",
            "",
            available=False,
        ),
        make_chunk(
            "international",
            "International Travel Award",
            "The maximum limit of the grant for the International Travel Award is INR 1.5 lakh.",
        ),
    ]
    index = HybridIndex(chunks)

    results = index.search("What is the maximum limit for the National Travel Award?", top_k=2)

    assert results[0].chunk.document_id == "national"


def test_quantity_answer_selects_number_with_matching_unit() -> None:
    chunks = [
        make_chunk(
            "drc",
            "DRC Guidelines",
            (
                "The term of DRC members is two years. "
                "The DRC consists of the HOD and 2 to 6 faculty members who are active in research."
            ),
        )
    ]
    results = HybridIndex(chunks).search("How many members are in the DRC?", top_k=1)

    answer = extractive_answer("How many members are in the DRC?", results)

    assert "2 to 6 faculty members" in answer.answer
    assert "two years" not in answer.answer


def test_unrelated_query_abstains() -> None:
    chunks = [make_chunk("drc", "DRC Guidelines", "The committee has two members.")]
    results = HybridIndex(chunks).search("What is today's cafeteria menu?", top_k=3)

    answer = extractive_answer("What is today's cafeteria menu?", results)

    assert answer.abstained
    assert answer.confidence == "low"


def test_results_include_score_explanation() -> None:
    chunks = [make_chunk("leave", "Leave Rules", "Casual leave is limited to 15 days.")]
    result = HybridIndex(chunks).search("How many casual leave days are allowed?", top_k=1)[0]

    assert result.bm25_score > 0
    assert result.tfidf_score > 0
    assert "leave" in result.matched_terms
    assert result.rank == 1


def test_institute_fellow_query_penalizes_self_sponsored_form() -> None:
    chunks = [
        make_chunk(
            "self",
            "Contingency Self Project",
            "Institute form for items purchased under a self sponsored PhD fellowship.",
        ),
        make_chunk(
            "institute",
            "Institute Contingency Form",
            "Details of items purchased under the contingency grant for Institute PhD fellows.",
        ),
    ]

    results = HybridIndex(chunks).search(
        "Which items can be purchased with an Institute PhD fellowship contingency?",
        top_k=2,
    )

    assert results[0].chunk.document_id == "institute"
