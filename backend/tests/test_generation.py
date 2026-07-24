import json

import httpx
import pytest

from backend.ir_system.generation import GeminiGenerator, GenerationValidationError
from backend.ir_system.models import Chunk, SearchResult
from backend.ir_system.service import SearchService


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}


def search_result() -> SearchResult:
    chunk = Chunk(
        chunk_id="one",
        document_id="one",
        file_name="policy.pdf",
        title="Policy",
        page_start=2,
        page_end=2,
        text="The maximum grant is ₹25,000.",
    )
    return SearchResult(
        chunk=chunk,
        score=0.9,
        bm25_score=1.0,
        tfidf_score=0.8,
        phrase_score=0.5,
        proximity_score=0.5,
        title_score=0.5,
        query_coverage=1.0,
        matched_terms=["grant"],
        rank=1,
    )


def test_gemini_answer_requires_a_valid_citation(monkeypatch) -> None:
    payload = {"status": "answered", "answer": "₹25,000", "citations": []}
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: FakeResponse(json.dumps(payload))
    )

    with pytest.raises(GenerationValidationError, match="inline citations"):
        GeminiGenerator("key", "model").generate("What is the grant?", [search_result()], "high")


def test_gemini_answer_accepts_and_records_valid_citations(monkeypatch) -> None:
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse(
            json.dumps(
                {
                    "status": "answered",
                    "answer": "The maximum grant is ₹25,000. [1]",
                    "citations": [1],
                }
            )
        )

    monkeypatch.setattr(
        httpx,
        "post",
        fake_post,
    )

    answer = GeminiGenerator("key", "model").generate(
        "What is the grant?", [search_result()], "high"
    )

    assert answer.answer_type == "gemini"
    assert answer.citations == [1]
    assert captured["headers"] == {"x-goog-api-key": "key"}
    assert "params" not in captured
    assert captured["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_cannot_cite_a_result_that_was_not_in_context(monkeypatch) -> None:
    results = [search_result()]
    for rank in range(2, 6):
        result = search_result()
        results.append(
            SearchResult(
                **{**result.__dict__, "rank": rank},
            )
        )
    monkeypatch.setattr(
        httpx,
        "post", lambda *args, **kwargs: FakeResponse(
            json.dumps(
                {
                    "status": "answered",
                    "answer": "Unsupported claim. [5]",
                    "citations": [5],
                }
            )
        ),
    )

    with pytest.raises(GenerationValidationError, match="valid source citations"):
        GeminiGenerator("key", "model").generate("Question?", results, "medium")


def test_gemini_no_answer_response_is_marked_as_abstained(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post", lambda *args, **kwargs: FakeResponse(
            json.dumps(
                {
                    "status": "not_found",
                    "answer": "The supplied sources do not state the requested rule.",
                    "citations": [],
                }
            )
        ),
    )

    answer = GeminiGenerator("key", "model").generate(
        "What is the rule?", [search_result()], "medium"
    )

    assert answer.abstained
    assert answer.confidence == "low"


@pytest.mark.parametrize(
    "refusal",
    [
        "I cannot answer this from the supplied context.",
        "There is not enough information in the available documents to determine this.",
    ],
)
def test_gemini_citationless_refusals_are_safe_abstentions(monkeypatch, refusal: str) -> None:
    monkeypatch.setattr(
        httpx,
        "post", lambda *args, **kwargs: FakeResponse(
            json.dumps({"status": "not_found", "answer": refusal, "citations": []})
        ),
    )

    answer = GeminiGenerator("key", "model").generate(
        "What is the missing rule?", [search_result()], "high"
    )

    assert answer.abstained
    assert answer.confidence == "low"
    assert answer.citations == []


def test_supported_negative_fact_is_not_mistaken_for_a_refusal(monkeypatch) -> None:
    payload = {
        "status": "answered",
        "answer": "The documents do not support clubbing casual leave with other leave. [1]",
        "citations": [1],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: FakeResponse(json.dumps(payload))
    )

    answer = GeminiGenerator("key", "model").generate(
        "Can casual leave be clubbed?", [search_result()], "high"
    )

    assert not answer.abstained
    assert answer.confidence == "high"


def test_structured_answer_rejects_ungrounded_numeric_claim(monkeypatch) -> None:
    payload = {
        "status": "answered",
        "answer": "The maximum grant is ₹99,000. [1]",
        "citations": [1],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: FakeResponse(json.dumps(payload))
    )

    with pytest.raises(GenerationValidationError, match="numeric claims"):
        GeminiGenerator("key", "model").generate(
            "What is the maximum grant?", [search_result()], "high"
        )


def test_answered_status_cannot_contain_an_unavailable_answer(monkeypatch) -> None:
    payload = {
        "status": "answered",
        "answer": "The answer is not available in the supplied sources. [1]",
        "citations": [1],
    }
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: FakeResponse(json.dumps(payload))
    )

    with pytest.raises(GenerationValidationError, match="contradicts"):
        GeminiGenerator("key", "model").generate(
            "What is the missing answer?", [search_result()], "high"
        )


def test_unstructured_provider_text_fails_grounding_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            "The answer is not available in the supplied sources. [1]"
        ),
    )

    with pytest.raises(GenerationValidationError, match="structured JSON"):
        GeminiGenerator("key", "model").generate(
            "What is the missing rule?", [search_result()], "high"
        )


def test_service_fails_closed_for_invalid_generated_output() -> None:
    class FakeIndex:
        corpus_terms = frozenset({"grant", "maximum"})
        corpus_text = "the maximum grant is ₹25,000"

        def search(self, query: str, *, top_k: int):
            return [search_result()]

    class InvalidGenerator:
        def generate(self, *args, **kwargs):
            raise GenerationValidationError("invalid structure")

    service = SearchService()
    service._index = FakeIndex()
    service._generator = InvalidGenerator()

    result = service.query("What is the maximum grant?", answer_mode="gemini")

    assert result.answer.abstained
    assert result.answer.answer_type == "gemini"
    assert result.answer.citations == []
    assert any("invalid citation structure" in warning for warning in result.warnings)
