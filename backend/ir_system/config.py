"""Central configuration and repository-relative paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cors_origins() -> Tuple[str, ...]:
    configured = os.getenv("CORS_ORIGINS", "")
    if configured.strip():
        return tuple(origin.strip() for origin in configured.split(",") if origin.strip())
    return ("http://localhost:3000", "http://127.0.0.1:3000")


@dataclass(frozen=True)
class Settings:
    """Runtime settings with conservative, local-first defaults."""

    corpus_dir: Path = PROJECT_ROOT / "document_corpus"
    ocr_dir: Path = PROJECT_ROOT / "data" / "ocr"
    evaluation_dataset: Path = PROJECT_ROOT / "data" / "evaluation" / "queries.csv"
    unanswerable_dataset: Path = PROJECT_ROOT / "data" / "evaluation" / "unanswerable.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    chunk_size: int = 900
    chunk_overlap: int = 140
    default_top_k: int = 5
    max_top_k: int = 10
    max_query_length: int = 500
    max_chunks_per_document: int = 2
    ocr_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_OCR", False))
    ocr_language: str = field(default_factory=lambda: os.getenv("OCR_LANGUAGE", "eng"))
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
    )
    cors_origins: Tuple[str, ...] = field(default_factory=_cors_origins)


settings = Settings()
