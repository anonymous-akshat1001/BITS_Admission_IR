"""Explainable local hybrid ranking for the small policy-document corpus."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from .models import Chunk, SearchResult
from .preprocessing import explicit_identifier_terms, query_terms, tokenize


FOCUS_TERMS = {
    "after", "before", "cgpa", "da", "fulltime", "id", "institute", "international", "less", "more",
    "national", "onduty", "parttime", "self", "special", "ta", "transfer",
}
FORM_TERMS = {"application", "form", "forms", "format", "proforma", "requisition"}
CONFLICTING_TERMS = {
    "institute": {"self"},
    "international": {"national"},
    "national": {"international"},
}


class HybridIndex:
    """Combine BM25, TF-IDF, phrase, proximity, and title evidence.

    The corpus is small enough to keep this transparent index in memory. It avoids a cloud
    database and model download while retaining the useful idea behind hybrid retrieval:
    rank candidates using signals with different strengths, then fuse them deterministically.
    """

    METHOD_NAME = "BM25 + TF-IDF + phrase/title reranking"

    def __init__(self, chunks: Sequence[Chunk], *, max_chunks_per_document: int = 2):
        if not chunks:
            raise ValueError("Cannot build a search index without chunks")
        self.chunks = list(chunks)
        self.max_chunks_per_document = max_chunks_per_document
        self.term_frequencies: List[Counter[str]] = []
        self.content_tokens: List[List[str]] = []
        self.title_token_sets: List[set[str]] = []
        self.document_lengths: List[int] = []
        self.document_frequency: Counter[str] = Counter()
        self.postings: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)

        for index, chunk in enumerate(self.chunks):
            content_tokens = tokenize(chunk.text)
            title_tokens = tokenize(chunk.title)
            section_tokens = tokenize(chunk.section)
            # Repeating title terms is a simple field boost, not hidden training.
            indexed_tokens = content_tokens + section_tokens + title_tokens + title_tokens
            frequencies = Counter(indexed_tokens)
            self.term_frequencies.append(frequencies)
            self.content_tokens.append(content_tokens)
            self.title_token_sets.append(set(title_tokens))
            self.document_lengths.append(max(1, len(indexed_tokens)))
            self.document_frequency.update(frequencies.keys())
            for term, frequency in frequencies.items():
                self.postings[term].append((index, frequency))

        self.document_count = len(self.chunks)
        self.average_document_length = sum(self.document_lengths) / self.document_count
        self.idf = {
            term: math.log((self.document_count + 1) / (frequency + 1)) + 1.0
            for term, frequency in self.document_frequency.items()
        }
        self.corpus_terms = frozenset(self.document_frequency)
        self.corpus_text = " ".join(
            " ".join(chunk.index_text.lower().split()) for chunk in self.chunks
        )
        self.vector_norms = [self._document_vector_norm(frequencies) for frequencies in self.term_frequencies]

    def _document_vector_norm(self, frequencies: Counter[str]) -> float:
        return math.sqrt(
            sum(((1.0 + math.log(frequency)) * self.idf[term]) ** 2 for term, frequency in frequencies.items())
        ) or 1.0

    def _bm25_scores(self, weighted_terms: Dict[str, float], k1: float = 1.5, b: float = 0.75) -> List[float]:
        scores = [0.0] * self.document_count
        for term, query_weight in weighted_terms.items():
            df = self.document_frequency.get(term, 0)
            if not df:
                continue
            inverse_document_frequency = math.log(
                1.0 + (self.document_count - df + 0.5) / (df + 0.5)
            )
            for document_index, frequency in self.postings[term]:
                length_ratio = self.document_lengths[document_index] / self.average_document_length
                denominator = frequency + k1 * (1.0 - b + b * length_ratio)
                scores[document_index] += (
                    query_weight
                    * inverse_document_frequency
                    * (frequency * (k1 + 1.0) / denominator)
                )
        return scores

    def _tfidf_scores(self, weighted_terms: Dict[str, float]) -> List[float]:
        query_weights = {
            term: weight * self.idf.get(term, 0.0) for term, weight in weighted_terms.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values())) or 1.0
        scores = [0.0] * self.document_count
        for term, query_weight in query_weights.items():
            if not query_weight:
                continue
            idf = self.idf.get(term, 0.0)
            for document_index, frequency in self.postings.get(term, ()):  # Sparse dot product.
                document_weight = (1.0 + math.log(frequency)) * idf
                scores[document_index] += query_weight * document_weight
        return [
            score / (query_norm * self.vector_norms[index])
            for index, score in enumerate(scores)
        ]

    @staticmethod
    def _phrase_score(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        if len(query_tokens) == 1:
            return float(query_tokens[0] in document_tokens)
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))
        document_bigrams = set(zip(document_tokens, document_tokens[1:]))
        return len(query_bigrams & document_bigrams) / max(1, len(query_bigrams))

    @staticmethod
    def _proximity_score(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        targets = set(query_tokens)
        if len(targets) < 2 or not document_tokens:
            return 0.0
        counts: Counter[str] = Counter()
        left = 0
        best_span: int | None = None
        for right, token in enumerate(document_tokens):
            if token in targets:
                counts[token] += 1
            while len(counts) == len(targets) and left <= right:
                span = right - left + 1
                best_span = span if best_span is None else min(best_span, span)
                left_token = document_tokens[left]
                if left_token in targets:
                    counts[left_token] -= 1
                    if counts[left_token] == 0:
                        del counts[left_token]
                left += 1
        if best_span is None:
            present = len(targets & set(document_tokens))
            return 0.25 * present / len(targets)
        return min(1.0, len(targets) / best_span)

    @staticmethod
    def _safe_normalize(values: Sequence[float]) -> List[float]:
        maximum = max(values, default=0.0)
        if maximum <= 0:
            return [0.0] * len(values)
        return [value / maximum for value in values]

    @staticmethod
    def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
        left_set, right_set = set(left), set(right)
        union = left_set | right_set
        return len(left_set & right_set) / len(union) if union else 0.0

    def search(self, query: str, *, top_k: int = 5) -> List[SearchResult]:
        original_terms, weighted_terms = query_terms(query)
        if not original_terms:
            return []

        bm25_raw = self._bm25_scores(weighted_terms)
        tfidf_raw = self._tfidf_scores(weighted_terms)
        bm25 = self._safe_normalize(bm25_raw)
        tfidf = self._safe_normalize(tfidf_raw)
        original_term_set = set(original_terms)
        focus_terms = (original_term_set & FOCUS_TERMS) | set(explicit_identifier_terms(query))

        candidates: List[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            indexed_term_set = set(self.term_frequencies[index])
            matched = original_term_set & indexed_term_set
            if not matched and bm25[index] <= 0 and tfidf[index] <= 0:
                continue
            coverage = len(matched) / max(1, len(original_term_set))
            title_score = len(original_term_set & self.title_token_sets[index]) / max(
                1, len(original_term_set)
            )
            phrase_score = self._phrase_score(original_terms, self.content_tokens[index])
            proximity_score = self._proximity_score(original_terms, self.content_tokens[index])
            score = (
                0.47 * bm25[index]
                + 0.25 * tfidf[index]
                + 0.10 * phrase_score
                + 0.08 * proximity_score
                + 0.10 * title_score
            )
            # Prefer evidence that covers several explicit query concepts, particularly for
            # near-duplicate questions such as "above" versus "below" ten percent.
            score *= 0.82 + 0.18 * coverage
            if focus_terms:
                focus_coverage = len(focus_terms & indexed_term_set) / len(focus_terms)
                score *= 0.35 + 0.65 * focus_coverage
            title_terms = set(tokenize(chunk.title))
            conflicting = {
                conflict
                for term in original_term_set
                for conflict in CONFLICTING_TERMS.get(term, set())
                if conflict not in original_term_set
            }
            if conflicting & indexed_term_set:
                score *= 0.55
            # A proposal and a thesis are different stages in this corpus. Avoid ranking a
            # proposal template above thesis-submission guidance (and vice versa) merely
            # because both contain generic PhD vocabulary.
            if (
                "thesis" in original_term_set
                and "proposal" in title_terms
                and "thesis" not in title_terms
            ) or (
                "proposal" in original_term_set
                and "thesis" in title_terms
                and "proposal" not in title_terms
            ):
                score *= 0.65
            form_probe = f"{chunk.section} {chunk.text[:260]}".lower()
            looks_like_form = bool(FORM_TERMS & title_terms) or bool(
                re.search(r"\b(?:application|checklist|format|proforma)\s+for\b", form_probe)
            )
            if looks_like_form and not (FORM_TERMS & original_term_set):
                score *= 0.78
            candidates.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    bm25_score=bm25[index],
                    tfidf_score=tfidf[index],
                    phrase_score=phrase_score,
                    proximity_score=proximity_score,
                    title_score=title_score,
                    query_coverage=coverage,
                    matched_terms=sorted(matched),
                )
            )

        candidates.sort(
            key=lambda item: (
                item.score,
                item.query_coverage,
                item.phrase_score,
                item.chunk.text_available,
                -item.chunk.page_start,
            ),
            reverse=True,
        )
        selected: List[SearchResult] = []
        per_document: Counter[str] = Counter()
        selected_token_sets: List[set[str]] = []
        deferred: List[SearchResult] = []
        for candidate in candidates:
            if candidate.score <= 0:
                continue
            document_id = candidate.chunk.document_id
            tokens = set(tokenize(candidate.chunk.text))
            too_similar = any(self._jaccard(tokens, existing) > 0.82 for existing in selected_token_sets)
            if per_document[document_id] >= self.max_chunks_per_document or too_similar:
                deferred.append(candidate)
                continue
            selected.append(candidate)
            selected_token_sets.append(tokens)
            per_document[document_id] += 1
            if len(selected) >= top_k:
                break

        # A narrow query may only match one document. Fill remaining places rather than return
        # fewer results, but retain overlap suppression in the normal path.
        if len(selected) < top_k:
            for candidate in deferred:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= top_k:
                    break

        return [replace(result, rank=rank) for rank, result in enumerate(selected[:top_k], start=1)]
