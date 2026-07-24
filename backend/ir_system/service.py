"""Thread-safe orchestration of ingestion, retrieval, and answer generation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Literal

from .answering import confidence_for, extractive_answer
from .config import Settings, settings
from .corpus import load_corpus
from .generation import GeminiGenerator, GenerationValidationError
from .models import AnswerResult, CorpusSummary, QueryResult
from .retrieval import HybridIndex


logger = logging.getLogger(__name__)
AnswerMode = Literal["auto", "extractive", "gemini"]


class SearchService:
    """Own the in-memory index and expose one consistent query operation."""

    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings
        self._index: HybridIndex | None = None
        self._summary: CorpusSummary | None = None
        self._initialization_error: str | None = None
        self._lock = threading.Lock()
        self._generator = (
            GeminiGenerator(
                app_settings.gemini_api_key,
                app_settings.gemini_model,
                app_settings.gemini_timeout_seconds,
            )
            if app_settings.gemini_api_key
            else None
        )

    @property
    def ready(self) -> bool:
        return self._index is not None

    @property
    def summary(self) -> CorpusSummary | None:
        return self._summary

    def status(self) -> Dict[str, object]:
        return {
            "status": (
                "degraded"
                if self._initialization_error is not None
                else "ready" if self.ready else "starting"
            ),
            "index_ready": self.ready,
            "answer_mode": "gemini + extractive fallback" if self._generator else "extractive",
            "initialization_error": (
                "Index initialization failed" if self._initialization_error else None
            ),
            "corpus": self._summary.to_dict() if self._summary else None,
        }

    def ensure_ready(self) -> None:
        if self._index is not None:
            return
        with self._lock:
            if self._index is not None:
                return
            try:
                chunks, summary = load_corpus(
                    self.settings.corpus_dir,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    enable_ocr=self.settings.ocr_enabled,
                    ocr_language=self.settings.ocr_language,
                    ocr_dir=self.settings.ocr_dir,
                )
                self._index = HybridIndex(
                    chunks,
                    max_chunks_per_document=self.settings.max_chunks_per_document,
                )
                self._summary = summary
                self._initialization_error = None
                logger.info(
                    "Local index ready: %d documents, %d pages, %d chunks",
                    summary.document_count,
                    summary.page_count,
                    summary.chunk_count,
                )
            except Exception as exc:
                self._initialization_error = str(exc)
                logger.exception("Failed to initialize the local search index")
                raise

    def query(self, query: str, *, top_k: int | None = None, answer_mode: AnswerMode = "auto") -> QueryResult:
        cleaned_query = " ".join((query or "").split())
        if not cleaned_query:
            raise ValueError("Query must not be empty")
        if len(cleaned_query) > self.settings.max_query_length:
            raise ValueError(f"Query must be at most {self.settings.max_query_length} characters")
        if answer_mode not in {"auto", "extractive", "gemini"}:
            raise ValueError("answer_mode must be auto, extractive, or gemini")
        result_count = top_k or self.settings.default_top_k
        if not 1 <= result_count <= self.settings.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.settings.max_top_k}")

        started = time.perf_counter()
        self.ensure_ready()
        assert self._index is not None
        results = self._index.search(cleaned_query, top_k=result_count)
        extractive = extractive_answer(
            cleaned_query,
            results,
            corpus_terms=self._index.corpus_terms,
            corpus_text=self._index.corpus_text,
        )
        answer = extractive
        warnings = []

        should_generate = answer_mode == "gemini" or (answer_mode == "auto" and self._generator is not None)
        if should_generate and not extractive.abstained:
            if self._generator is None:
                warnings.append("Gemini was requested but GEMINI_API_KEY is not configured; using extractive mode.")
            else:
                try:
                    answer = self._generator.generate(
                        cleaned_query, results, confidence_for(results)
                    )
                except GenerationValidationError as exc:
                    logger.warning(
                        "Gemini output failed grounding validation (%s)",
                        type(exc).__name__,
                    )
                    answer = AnswerResult(
                        answer=(
                            "The optional generator did not return a verifiable grounded answer, "
                            "so no generated answer is shown."
                        ),
                        answer_type="gemini",
                        confidence="low",
                        citations=[],
                        abstained=True,
                    )
                    warnings.append(
                        "Optional answer generation returned an invalid citation structure; the response was refused."
                    )
                except Exception as exc:
                    # Do not stringify HTTP errors: request metadata can contain credentials.
                    logger.warning(
                        "Gemini generation failed; using extractive fallback (%s)",
                        type(exc).__name__,
                    )
                    warnings.append("Optional answer generation failed; the local extractive answer is shown instead.")
        if extractive.warning:
            warnings.append(extractive.warning)
        if self._summary and self._summary.scanned_or_empty_documents:
            matched_scans = [
                result.chunk.file_name for result in results if not result.chunk.text_available
            ]
            if matched_scans and not extractive.warning:
                warnings.append("One retrieved PDF has no searchable text and may require OCR.")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QueryResult(
            query=cleaned_query,
            answer=answer,
            results=results,
            processing_time_ms=elapsed_ms,
            retrieval_method=HybridIndex.METHOD_NAME,
            warnings=warnings,
        )
