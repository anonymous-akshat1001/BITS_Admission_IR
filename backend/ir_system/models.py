"""Small data models shared by ingestion, retrieval, and answering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    file_name: str
    title: str
    page_start: int
    page_end: int
    text: str
    section: str = ""
    text_available: bool = True

    @property
    def index_text(self) -> str:
        parts = [self.title, self.section, self.text]
        return "\n".join(part for part in parts if part)


@dataclass
class CorpusSummary:
    document_count: int = 0
    page_count: int = 0
    chunk_count: int = 0
    searchable_document_count: int = 0
    scanned_or_empty_documents: List[str] = field(default_factory=list)
    ocr_sidecar_documents: List[str] = field(default_factory=list)
    ocr_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    bm25_score: float
    tfidf_score: float
    phrase_score: float
    proximity_score: float
    title_score: float
    query_coverage: float
    matched_terms: List[str]
    rank: int = 0


@dataclass
class AnswerResult:
    answer: str
    answer_type: str
    confidence: str
    citations: List[int]
    abstained: bool = False
    warning: str | None = None


@dataclass
class QueryResult:
    query: str
    answer: AnswerResult
    results: List[SearchResult]
    processing_time_ms: float
    retrieval_method: str
    warnings: List[str] = field(default_factory=list)
