"""Dependency-free retrieval, answer-overlap, and timing metrics."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence, Set


CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def unique_documents(document_ids: Iterable[str]) -> List[str]:
    """Keep the first (best) occurrence of every document identifier."""

    return list(dict.fromkeys(document_id for document_id in document_ids if document_id))


def hit_at_k(ranked_documents: Sequence[str], relevant_documents: Set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return float(bool(set(ranked_documents[:k]) & relevant_documents))


def reciprocal_rank(
    ranked_documents: Sequence[str],
    relevant_documents: Set[str],
    *,
    cutoff: int = 10,
) -> float:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    for rank, document_id in enumerate(ranked_documents[:cutoff], start=1):
        if document_id in relevant_documents:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_documents: Sequence[str], relevant_documents: Set[str], k: int) -> float:
    """Binary nDCG for document-level qrels."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_documents:
        return 0.0
    gains = [1.0 if document_id in relevant_documents else 0.0 for document_id in ranked_documents[:k]]
    dcg = sum(gain / math.log2(rank + 1.0) for rank, gain in enumerate(gains, start=1))
    ideal_relevant = min(k, len(relevant_documents))
    ideal_dcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_relevant + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def answer_tokens(text: str) -> List[str]:
    """Normalize answer text while ignoring UI citation markers."""

    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = CITATION_RE.sub(" ", normalized).casefold()
    normalized = normalized.encode("ascii", errors="ignore").decode("ascii")
    return TOKEN_RE.findall(normalized)


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = answer_tokens(reference)
    prediction_tokens = answer_tokens(prediction)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return LCS length using O(min(n, m)) memory."""

    if len(left) < len(right):
        shorter, longer = left, right
    else:
        shorter, longer = right, left
    previous = [0] * (len(shorter) + 1)
    for long_token in longer:
        current = [0]
        for index, short_token in enumerate(shorter, start=1):
            if long_token == short_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = answer_tokens(reference)
    prediction_tokens = answer_tokens(prediction)
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    if not lcs:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linearly interpolate a quantile on the inclusive [0, 1] scale."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

