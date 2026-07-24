"""Reproducible local proxy for the unavailable original Qdrant system.

The historical repository did not include its Qdrant index, credentials, or saved
evaluation output.  This module therefore preserves the most visible legacy text
choices (whole-document cleanup, 600-character chunks, and a top-chunk answer)
while replacing the unavailable vector services with transparent unigram TF-IDF.
It is a comparison proxy, not a claim about the original system's measured scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class BaselineChunk:
    """One fixed-width chunk in the proxy baseline."""

    chunk_id: str
    file_name: str
    text: str
    position: int


@dataclass(frozen=True)
class BaselineResult:
    """A ranked baseline chunk and its cosine score."""

    chunk: BaselineChunk
    score: float
    rank: int


def legacy_clean_text(text: str) -> str:
    """Apply the original pipeline's lossy document-level normalization."""

    lowered = (text or "").lower().replace("_", " ")
    ascii_text = lowered.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip()


def fixed_character_chunks(
    text: str,
    *,
    chunk_size: int = 600,
    overlap: int = 150,
) -> List[str]:
    """Split text into deterministic fixed-width windows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class LegacyTfidfBaseline:
    """Whole-document legacy preprocessing plus unigram TF-IDF cosine ranking."""

    METHOD_NAME = "legacy proxy: fixed chunks + unigram TF-IDF"

    def __init__(self, chunks: Sequence[BaselineChunk]):
        if not chunks:
            raise ValueError("Cannot build the baseline without searchable PDF text")
        self.chunks = list(chunks)
        self.term_frequencies: List[Counter[str]] = []
        self.document_frequency: Counter[str] = Counter()
        self.postings: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)

        for chunk_index, chunk in enumerate(self.chunks):
            frequencies = Counter(_tokens(chunk.text))
            self.term_frequencies.append(frequencies)
            self.document_frequency.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self.postings[term].append((chunk_index, frequency))

        chunk_count = len(self.chunks)
        self.idf: Dict[str, float] = {
            term: math.log((chunk_count + 1.0) / (frequency + 1.0)) + 1.0
            for term, frequency in self.document_frequency.items()
        }
        self.vector_norms = [self._vector_norm(frequencies) for frequencies in self.term_frequencies]

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Path,
        *,
        chunk_size: int = 600,
        overlap: int = 150,
    ) -> "LegacyTfidfBaseline":
        """Extract PDFs as whole documents and construct the proxy index."""

        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - dependency validation only.
            raise RuntimeError("PyMuPDF is required to build the evaluation baseline") from exc

        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        chunks: List[BaselineChunk] = []
        pdf_paths = sorted(corpus_dir.glob("*.pdf"), key=lambda path: path.name.lower())
        for pdf_path in pdf_paths:
            with fitz.open(pdf_path) as document:
                whole_document = "\n".join(page.get_text("text") for page in document)
            cleaned = legacy_clean_text(whole_document)
            for position, chunk_text in enumerate(
                fixed_character_chunks(cleaned, chunk_size=chunk_size, overlap=overlap),
                start=1,
            ):
                chunks.append(
                    BaselineChunk(
                        chunk_id=f"{pdf_path.name}::c{position}",
                        file_name=pdf_path.name,
                        text=chunk_text,
                        position=position,
                    )
                )
        return cls(chunks)

    def _vector_norm(self, frequencies: Counter[str]) -> float:
        squared = sum(
            (frequency * self.idf[term]) ** 2 for term, frequency in frequencies.items()
        )
        return math.sqrt(squared) or 1.0

    def search(self, query: str, *, top_k: int = 10) -> List[BaselineResult]:
        """Rank chunks by cosine similarity; deterministic zero-score ties remain ranked."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_frequencies = Counter(_tokens(legacy_clean_text(query)))
        query_weights = {
            term: frequency * self.idf[term]
            for term, frequency in query_frequencies.items()
            if term in self.idf
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values())) or 1.0
        dot_products = [0.0] * len(self.chunks)
        for term, query_weight in query_weights.items():
            idf = self.idf[term]
            for chunk_index, frequency in self.postings.get(term, ()):
                dot_products[chunk_index] += query_weight * frequency * idf

        scores = [
            dot_product / (query_norm * self.vector_norms[index])
            for index, dot_product in enumerate(dot_products)
        ]
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda index: (
                -scores[index],
                self.chunks[index].file_name.lower(),
                self.chunks[index].position,
            ),
        )
        return [
            BaselineResult(chunk=self.chunks[index], score=scores[index], rank=rank)
            for rank, index in enumerate(ranked_indices[:top_k], start=1)
        ]

    @staticmethod
    def answer(results: Sequence[BaselineResult]) -> str:
        """Match the legacy API behavior by returning the highest-ranked chunk."""

        return results[0].chunk.text if results else ""
