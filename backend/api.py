"""FastAPI application for local research-regulations search."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Path as ApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ir_system.config import settings
from .ir_system.models import QueryResult, SearchResult
from .ir_system.service import SearchService


logger = logging.getLogger(__name__)
service = SearchService(settings)


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=settings.max_query_length)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    answer_mode: Literal["auto", "extractive", "gemini"] = "auto"

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 2:
            raise ValueError("query must contain at least two visible characters")
        return cleaned


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_name: str
    title: str
    page_start: int
    page_end: int
    section: str = ""
    source_url: str
    text_available: bool


class SourceDocumentResponse(BaseModel):
    rank: int
    excerpt: str
    page_content: str
    score: float
    matched_terms: List[str]
    score_breakdown: dict[str, float]
    metadata: SourceMetadata


class QueryResponse(BaseModel):
    query: str
    answer: str
    answer_type: str
    confidence: str
    abstained: bool
    citations: List[int]
    retrieval_method: str
    processing_time_ms: float
    source_documents: List[SourceDocumentResponse]
    warnings: List[str]


def _source_response(result: SearchResult) -> SourceDocumentResponse:
    source_url = f"/documents/{quote(result.chunk.file_name, safe='')}"
    excerpt = result.chunk.text.strip()
    if len(excerpt) > 1000:
        excerpt = excerpt[:997].rsplit(" ", 1)[0] + "..."
    if not excerpt:
        excerpt = "No machine-readable text was extracted from this PDF."
    return SourceDocumentResponse(
        rank=result.rank,
        excerpt=excerpt,
        page_content=excerpt,
        score=round(result.score, 4),
        matched_terms=result.matched_terms,
        score_breakdown={
            "bm25": round(result.bm25_score, 4),
            "tfidf": round(result.tfidf_score, 4),
            "phrase": round(result.phrase_score, 4),
            "proximity": round(result.proximity_score, 4),
            "title": round(result.title_score, 4),
            "query_coverage": round(result.query_coverage, 4),
        },
        metadata=SourceMetadata(
            doc_name=result.chunk.file_name,
            title=result.chunk.title,
            page_start=result.chunk.page_start,
            page_end=result.chunk.page_end,
            section=result.chunk.section,
            source_url=source_url,
            text_available=result.chunk.text_available,
        ),
    )


def _query_response(result: QueryResult) -> QueryResponse:
    return QueryResponse(
        query=result.query,
        answer=result.answer.answer,
        answer_type=result.answer.answer_type,
        confidence=result.answer.confidence,
        abstained=result.answer.abstained,
        citations=result.answer.citations,
        retrieval_method=result.retrieval_method,
        processing_time_ms=round(result.processing_time_ms, 2),
        source_documents=[_source_response(item) for item in result.results],
        warnings=result.warnings,
    )


app = FastAPI(
    title="BITS Research Regulations Search API",
    description="Local, explainable retrieval and grounded answering over the bundled BITS research PDFs.",
    version="3.0.0",
)

allow_all_origins = settings.cors_origins == ("*",)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=not allow_all_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "name": "BITS Research Regulations Search API",
        "version": "3.0.0",
        "docs": "/docs",
        **service.status(),
    }


@app.get("/health")
async def health() -> dict[str, object]:
    try:
        await asyncio.to_thread(service.ensure_ready)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The document index could not be initialized.") from exc
    return service.status()


@app.get("/corpus")
async def corpus_summary() -> dict[str, object]:
    try:
        await asyncio.to_thread(service.ensure_ready)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The document index could not be initialized.") from exc
    return service.status()


@app.post("/query/", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        result = await asyncio.to_thread(
            service.query,
            request.query,
            top_k=request.top_k,
            answer_mode=request.answer_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search request failed")
        raise HTTPException(status_code=503, detail="The search index is temporarily unavailable.") from exc
    return _query_response(result)


# Compatibility aliases allow older clients to keep working while using the improved ranker.
@app.post("/query/hybrid/", response_model=QueryResponse, include_in_schema=False)
async def legacy_hybrid_query(request: QueryRequest) -> QueryResponse:
    return await query(request)


@app.post("/query/hybrid-rerank/", response_model=QueryResponse, include_in_schema=False)
async def legacy_rerank_query(request: QueryRequest) -> QueryResponse:
    return await query(request)


@app.get("/documents/{file_name}", response_class=FileResponse)
async def document(
    file_name: str = ApiPath(min_length=5, max_length=240),
) -> FileResponse:
    # Only a basename may be requested; resolve and verify the target remains inside the corpus.
    if Path(file_name).name != file_name or not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=404, detail="Document not found")
    target = (settings.corpus_dir / file_name).resolve()
    corpus_root = settings.corpus_dir.resolve()
    if target.parent != corpus_root or not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(
        target,
        media_type="application/pdf",
        filename=file_name,
        content_disposition_type="inline",
    )
