from fastapi.testclient import TestClient

from backend.api import app


client = TestClient(app)


def test_health_does_not_require_cloud_credentials() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["index_ready"] is True
    assert response.json()["answer_mode"] in {"extractive", "gemini + extractive fallback"}


def test_query_returns_page_citations_and_scores() -> None:
    response = client.post(
        "/query/",
        json={
            "query": "How many credits are prescribed for a PhD student with a first degree?",
            "top_k": 3,
            "answer_mode": "extractive",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "24 units" in payload["answer"]
    assert payload["source_documents"]
    source = payload["source_documents"][0]
    assert source["metadata"]["page_start"] >= 1
    assert source["metadata"]["source_url"].startswith("/documents/")
    assert source["score_breakdown"]["bm25"] >= 0
    assert 0 <= source["score_breakdown"]["query_coverage"] <= 1
    assert "file_path" not in source["metadata"]


def test_query_rejects_blank_input() -> None:
    response = client.post("/query/", json={"query": "   "})
    assert response.status_code == 422


def test_document_route_rejects_path_traversal() -> None:
    response = client.get("/documents/..%2FREADME.md")
    assert response.status_code == 404
