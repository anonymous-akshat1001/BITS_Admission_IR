"""Unicode-safe text cleanup and domain-aware query preprocessing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


# Question words and very common function words add noise to lexical ranking. Important
# operators such as "before", "after", "less", "more", and negations are retained.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "case",
    "could", "did", "do", "does", "e.g", "for", "from", "get", "got", "had", "has", "have", "how",
    "i", "in", "into", "is", "it", "its", "many", "may", "much", "my", "of", "on", "or", "our",
    "please", "present", "say", "should", "so", "some", "something", "than", "that", "the",
    "their", "them", "there", "these", "they", "this", "those", "to", "under",
    "up", "was", "we", "were", "what", "when", "where", "which", "who", "why",
    "s", "sir", "will", "with", "would", "you", "your", "tell", "about", "give",
}

DOMAIN_TERMS = {
    "admission", "agsrd", "award", "budget", "consultancy", "contingency", "dac", "doctoral",
    "da", "drc", "endorsement", "faculty", "fellowship", "gcir", "grant", "leave", "overhead", "phd",
    "pi", "professionaldevelopmentfund", "departmentdevelopmentfund", "campusdevelopmentfund",
    "project", "proposal", "publication", "qualifying", "reimbursement", "research", "scholar",
    "stipend", "supervisor", "ta", "thesis", "travel",
}

# A slash usually separates alternatives/acronyms in policy text (TA/DA,
# international/national), so it is a boundary rather than part of one token.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", re.IGNORECASE)

CANONICAL_PHRASES: Sequence[Tuple[re.Pattern[str], str]] = (
    (
        re.compile(r"\bt\s*\.?\s*a\s*\.\s*d\s*\.?\s*a(?:\s*\.)?", re.I),
        "ta da",
    ),
    (
        re.compile(
            r"\bt\s*\.?\s*a\s*\.?(?:\s*[/&\-]\s*|\s+and\s+)"
            r"d\s*\.?\s*a(?:\s*\.)?",
            re.I,
        ),
        "ta da",
    ),
    (re.compile(r"\bph\s*\.?\s*d\.?\b", re.I), "phd"),
    (re.compile(r"\bfull[ -]?time\b", re.I), "fulltime"),
    (re.compile(r"\bpart[ -]?time\b", re.I), "parttime"),
    (re.compile(r"\bon[ -]?duty\b", re.I), "onduty"),
    (re.compile(r"\bswitch(?:\s+over)?\b", re.I), "transfer"),
    (re.compile(r"\bdepartment(?:al)? research committee\b", re.I), "drc"),
    (re.compile(r"\bdoctoral advisory committee\b", re.I), "dac"),
    (re.compile(r"\bacademic graduate studies\s*(?:and|&)\s*research division\b", re.I), "agsrd"),
    (re.compile(r"\bprofessional development fund\b", re.I), "professionaldevelopmentfund pdf"),
    (re.compile(r"\bdepartment development fund\b", re.I), "departmentdevelopmentfund ddf"),
    (re.compile(r"\bcampus development fund\b", re.I), "campusdevelopmentfund cdf"),
    (re.compile(r"\bprincipal investigator\b", re.I), "pi"),
    (re.compile(r"\bidentification(?:\s+number)?\b", re.I), "id"),
    (re.compile(r"\btravel allowance\b", re.I), "ta"),
    (re.compile(r"\bdaily allowance\b", re.I), "da"),
)

# Expansion is deliberately small and transparent. Added terms receive a lower weight than
# the user's own terms in the ranker.
QUERY_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "stipend": ("fellowship", "scholarship"),
    "student": ("scholar",),
    "hostel": ("accommodation",),
    "purchase": ("procure", "reimbursement"),
    "purchased": ("procure", "reimbursement"),
    "job": ("employment", "parttime"),
    "language": ("english",),
    "term": ("tenure", "duration"),
    "duration": ("semester", "time", "limit"),
    "credit": ("unit",),
    "members": ("composition",),
    "supervisor": ("supervision",),
    "proposal": ("research",),
    "award": ("grant",),
    "grant": ("award",),
    "documents": ("enclosures",),
    "steps": ("procedure", "process"),
    "ta": ("travel", "allowance", "reimbursement"),
    "da": ("daily", "allowance", "reimbursement"),
    "fil": ("submitted", "form"),
    "due": ("submitted", "within", "deadline"),
    "claim": ("reimbursement", "form"),
}

_STEM_EXCEPTIONS = {
    "admission", "admissions", "analysis", "bits", "business", "campus", "class",
    "consensus", "crisis", "davis", "fees", "guidelines", "news", "physics", "process",
    "research", "series", "status", "thesis", "this",
}


def normalize_unicode(text: str) -> str:
    """Normalize common PDF punctuation without discarding meaningful Unicode."""
    text = unicodedata.normalize("NFKC", text or "")
    return (
        text.replace("\u00ad", "")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2022", "•")
    )


def canonicalize_phrases(text: str) -> str:
    normalized = normalize_unicode(text).lower()
    for pattern, replacement in CANONICAL_PHRASES:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def light_stem(token: str) -> str:
    """Apply conservative suffix normalization suitable for this small policy corpus."""
    if token in _STEM_EXCEPTIONS or token.isdigit() or len(token) <= 4:
        return token
    irregular = {"criteria": "criterion", "indices": "index", "leaves": "leave", "policies": "policy"}
    if token in irregular:
        return irregular[token]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        base = token[:-3]
        if len(base) > 3 and base[-1:] == base[-2:-1]:
            base = base[:-1]
        return base
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("es") and len(token) > 5 and not token.endswith(("ses", "xes")):
        return token[:-2]
    if token.endswith("s") and len(token) > 5:
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> List[str]:
    canonical = canonicalize_phrases(text)
    tokens: List[str] = []
    for match in TOKEN_RE.finditer(canonical):
        raw_token = match.group(0).strip("./-")
        # Check before and after stemming: otherwise a stopword such as "something"
        # becomes "someth" and accidentally enters the index.
        if remove_stopwords and raw_token in STOPWORDS:
            continue
        token = light_stem(raw_token)
        if token and (not remove_stopwords or token not in STOPWORDS):
            tokens.append(token)
    return tokens


def explicit_identifier_terms(text: str) -> List[str]:
    """Return explicit all-cap/dotted identifiers such as ID, FCRA, QE, or T.A."""
    normalized = normalize_unicode(text)
    # Remove known compound notations before detecting unknown acronyms. For example,
    # T.A.D.A. represents the two supported concepts TA and DA, not an external TADA entity.
    for phrase_pattern, replacement in CANONICAL_PHRASES:
        normalized = phrase_pattern.sub(replacement, normalized)
    identifiers: List[str] = []
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Z](?:\s*\.\s*[A-Z])+\.?|[A-Z]{2,})(?![A-Za-z0-9])"
    )
    for match in pattern.finditer(normalized):
        compact = re.sub(r"[^A-Za-z0-9]", "", match.group(0)).lower()
        if len(compact) >= 2 and compact not in STOPWORDS:
            identifiers.append(light_stem(compact))
    return list(dict.fromkeys(identifiers))


def query_terms(query: str) -> Tuple[List[str], Dict[str, float]]:
    """Return original terms and weighted terms including modest domain expansion."""
    original = list(dict.fromkeys(tokenize(query)))
    weights: Dict[str, float] = {term: 1.0 for term in original}
    for term in original:
        for expansion in QUERY_EXPANSIONS.get(term, ()):
            normalized = light_stem(expansion)
            weights[normalized] = max(weights.get(normalized, 0.0), 0.35)
    return original, weights


def token_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def content_words(text: str) -> List[str]:
    return tokenize(text)


def has_domain_anchor(text: str) -> bool:
    return bool(set(tokenize(text)) & DOMAIN_TERMS)


def unique_in_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))
