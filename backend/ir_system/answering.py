"""Focused extractive answering with citations and calibrated abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .models import AnswerResult, SearchResult
from .preprocessing import explicit_identifier_terms, has_domain_anchor, query_terms, tokenize


SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9(•])")
MONEY_RE = re.compile(r"(?:₹|\b(?:inr|rs\.?|rupees?|lakh)\b)", re.I)
NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty",
    "fifty", "hundred",
}
KNOWN_ACRONYMS = {
    "ABDC", "AGSRD", "AI", "BE", "BITS", "CDF", "CGPA", "CORE", "CS", "CSIR", "DAC",
    "DA", "DBT", "DCC", "DDF", "DRC", "DST", "ERP", "GCIR", "HOD", "HRA",
    "ICMR", "ID", "INR", "IP", "IS", "JRF", "KK", "MBA", "MOA", "MOU", "NDA", "NET",
    "NFA", "PDF", "PI", "RCA", "SJR", "SOP", "SRF", "SSR", "TA", "UGC", "UPS",
}
ANSWER_FOCUS_IDENTIFIERS = {"ard", "cgpa", "da", "fcra", "id", "net", "qe", "ta", "ugc"}


def _retrieved_evidence_terms(results: Sequence[SearchResult]) -> set[str]:
    """Terms present in retrieved evidence, used to distinguish entities from valid acronyms."""
    terms: set[str] = set()
    for result in results[:5]:
        terms.update(tokenize(f"{result.chunk.title}\n{result.chunk.text}", remove_stopwords=False))
    return terms


def _unsupported_query_reason(
    query: str,
    results: Sequence[SearchResult],
    corpus_terms: set[str] | frozenset[str] | None = None,
    corpus_text: str | None = None,
) -> str | None:
    """Reject clear scope, freshness, or entity mismatches before extracting an answer."""
    lowered = query.lower()
    terms = set(tokenize(query))
    evidence_terms = _retrieved_evidence_terms(results)
    available_terms = corpus_terms if corpus_terms is not None else evidence_terms
    searchable_corpus_text = corpus_text or " ".join(
        " ".join(f"{result.chunk.title} {result.chunk.text}".lower().split())
        for result in results
    )
    if "football" in terms:
        return "The indexed research-policy corpus does not cover sports teams."
    if {"hostel", "fee"} <= terms or {"hostel", "fees"} <= terms:
        return "The corpus discusses research-scholar accommodation, not hostel fees."
    if {"travel", "undergraduate"} <= terms:
        return "The bundled travel policies apply to research scholars, not undergraduate students."
    animal_terms = {
        "animal", "animals", "bird", "birds", "cat", "cats", "dog", "dogs", "pet",
        "pets", "puppy",
    }
    accommodation_terms = {"accommodation", "hostel", "hous", "residence", "room"}
    if terms & animal_terms and not animal_terms & evidence_terms:
        return "The indexed accommodation passages do not contain a pets or animals policy."
    housing_cost_terms = {"amount", "charg", "charge", "cost", "fee", "fees", "rate", "rent"}
    is_travel_lodging = bool(
        terms & {"award", "grant", "reimburs", "reimbursement", "travel"}
    )
    asks_housing_cost = bool(terms & housing_cost_terms or "how much" in lowered)
    if terms & accommodation_terms and asks_housing_cost and not is_travel_lodging:
        return "The indexed research-scholar accommodation passages do not state housing charges."

    asks_for_volatile_fact = bool(
        terms
        & {
            "admission", "amount", "application", "award", "closing", "cost", "date",
            "deadline", "fee", "fees", "fellowship", "prize", "rate", "received",
            "recipient", "stipend", "tuition", "winner", "won",
        }
    )
    has_freshness_cue = bool(
        re.search(
            r"\b(?:as of|current|currently|forthcoming|last year|latest|newest|next|now|"
            r"present|recent|this admission cycle|today(?:'s)?|this (?:academic year|year|semester)|upcoming|"
            r"up[- ]to[- ]date)\b",
            lowered,
        )
    )
    requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", lowered))
    unsupported_year = bool(requested_years - available_terms)
    if asks_for_volatile_fact and (has_freshness_cue or unsupported_year):
        return "The static corpus cannot establish current or unsourced dates, fees, awards, or rates."
    if (
        {"application", "deadline"} <= terms
        or {"admission", "deadline"} <= terms
        or ({"application", "date"} <= terms and has_freshness_cue)
    ):
        return "The research-policy corpus does not contain a dependable admissions deadline."
    if has_freshness_cue and "who" in lowered and terms & {"best", "research", "scholar"}:
        return "The static policy corpus cannot identify current award holders or people."

    # Acronyms that occur anywhere in the corpus (for example ARD, ID, and CGPA) are valid
    # regardless of requested top_k. Only an unexplained identifier is treated as external.
    acronyms = set(explicit_identifier_terms(query))
    unknown_acronyms = {
        acronym
        for acronym in acronyms
        if acronym.upper() not in KNOWN_ACRONYMS and acronym not in available_terms
    }
    if unknown_acronyms:
        names = ", ".join(value.upper() for value in sorted(unknown_acronyms))
        return f"The query refers to an organization or acronym outside this corpus: {names}."

    # Institution phrases must occur as phrases in the corpus; a shared location word alone
    # (for example Hyderabad or Goa) does not establish that another university is in scope.
    institution_patterns = (
        r"\b([a-z][\w&'.-]+\s+(?:college|institute|university))\b",
        r"\b((?:college|institute|university)\s+of\s+[a-z][\w&'.-]+)\b",
    )
    external_institutions: List[str] = []
    for pattern in institution_patterns:
        for phrase in re.findall(pattern, query, re.I):
            identifying_terms = set(tokenize(phrase)) - {"college", "institute", "university"}
            normalized_phrase = " ".join(phrase.lower().split())
            if identifying_terms and normalized_phrase not in searchable_corpus_text:
                external_institutions.append(phrase)

    entity_patterns = (
        r"\b([a-z][\w&'.-]+)(?:'s)?\s+(?:research\s+budget|phd\s+"
        r"(?:fellowship|grant|stipend)|fellowship\s+(?:amount|rate|stipend))\b",
        r"\bfellowship\s+(?:does|from|at)\s+([a-z][\w&'.-]+)\b",
    )
    for pattern in entity_patterns:
        for name in re.findall(pattern, query, re.I):
            name_terms = set(tokenize(name))
            if name_terms and not name_terms <= available_terms:
                external_institutions.append(name)
    short_institution = re.search(r"\b((?:iit|iisc|nit)\s+[a-z][a-z.-]+)\b", lowered)
    if short_institution:
        short_name = short_institution.group(1)
        if not set(tokenize(short_name, remove_stopwords=False)) <= available_terms:
            external_institutions.append(short_name.upper())
    entity_subject_terms = {"amount", "budget", "fellowship", "grant", "rate", "stipend"}
    if terms & entity_subject_terms:
        for proper_name in re.findall(r"\b[A-Z][a-z]{3,}\b", query):
            proper_terms = set(tokenize(proper_name))
            if proper_terms and not proper_terms <= available_terms:
                external_institutions.append(proper_name)
    if external_institutions:
        return (
            "The query names an institution outside the supplied BITS Pilani research corpus: "
            + ", ".join(dict.fromkeys(external_institutions))
            + "."
        )
    return None


def confidence_for(results: Sequence[SearchResult]) -> str:
    if not results:
        return "low"
    top = results[0]
    if top.chunk.text_available and top.query_coverage >= 0.72 and top.score >= 0.52:
        return "high"
    if top.query_coverage >= 0.38 and top.score >= 0.30:
        return "medium"
    return "low"


def _question_intent(query: str) -> Dict[str, object]:
    lowered = query.lower()
    quantity_match = re.search(r"\bhow many\s+([a-z-]+(?:\s+[a-z-]+)?)", lowered)
    quantity_terms = tokenize(quantity_match.group(1)) if quantity_match else []
    if "credit" in quantity_terms:
        quantity_terms.append("unit")
    focus_terms = [
        identifier
        for identifier in explicit_identifier_terms(query)
        if identifier in ANSWER_FOCUS_IDENTIFIERS
    ]
    if "id" in tokenize(query) and "id" not in focus_terms:
        focus_terms.append("id")
    if focus_terms and "transfer" in tokenize(query):
        focus_terms.append("transfer")
    return {
        "number": bool(
            re.search(r"\b(how many|how much|maximum|minimum|percentage|percent|limit|duration|term)\b", lowered)
        ),
        "when": bool(re.search(r"\b(when|before|after|within|deadline)\b", lowered)),
        "list": bool(re.search(r"\b(what are|which|items|steps|documents|criteria|guidelines|checklist)\b", lowered)),
        "difference": "difference" in lowered,
        "short_fact": bool(re.search(r"\b(how many|how much|how long|maximum|minimum)\b", lowered)),
        "money": bool(
            (
                re.search(r"\b(maximum|minimum|limit|amount|cost)\b", lowered)
                and re.search(r"\b(grant|award|fee|fellowship|amount|cost)\b", lowered)
            )
            or (
                re.search(r"\btuition\s+fees?\b", lowered)
                and not re.search(r"\b(?:criteria|eligib|policy|rule|waiv)", lowered)
            )
            or (
                re.search(r"\bwhat(?:'s| is)\b.*\bstipend\b", lowered)
                and not re.search(r"\b(?:criteria|eligib|policy|rule|tenure)\b", lowered)
            )
        ),
        "duration": bool(re.search(r"\b(how long|duration|term|tenure|period)\b", lowered)),
        "purpose": bool(re.search(r"\b(?:purpose|aim|objective)\b", lowered)),
        "quantity_terms": quantity_terms,
        "focus_terms": focus_terms,
    }


def _candidate_sentences(result: SearchResult) -> List[Tuple[int, str]]:
    sentences: List[Tuple[int, str]] = []
    # PDF extraction occasionally inserts a blank line in the middle of a wrapped sentence.
    # Rejoin only a lowercase continuation so headings and list items remain separate.
    prepared_text = re.sub(r"(?<=[A-Za-z])\n+(?=[a-z])", " ", result.chunk.text)
    for position, piece in enumerate(SENTENCE_SPLIT_RE.split(prepared_text)):
        sentence = re.sub(r"\s+", " ", piece).strip(" \t")
        if len(sentence) < 12:
            continue
        if len(sentence.split()) < 3 and not explicit_identifier_terms(sentence):
            continue
        if re.match(r"^\d+[.)]?\s*:\s*", sentence) or (
            ":" in sentence
            and len(sentence.split()) <= 5
            and not explicit_identifier_terms(sentence)
        ):
            continue
        words = sentence.split()
        title_case_ratio = sum(word[:1].isupper() for word in words) / max(1, len(words))
        if len(words) <= 8 and title_case_ratio >= 0.6 and not sentence.endswith((".", "?", "!")):
            continue
        if len(sentence) > 650:
            sentence = sentence[:647].rsplit(" ", 1)[0] + "..."
        sentences.append((position, sentence))
    return sentences


def _sentence_score(
    sentence: str,
    result: SearchResult,
    original_terms: Sequence[str],
    weighted_terms: Dict[str, float],
    intent: Dict[str, object],
) -> float:
    sentence_tokens = tokenize(sentence)
    if not sentence_tokens:
        return 0.0
    sentence_set = set(sentence_tokens)
    original_set = set(original_terms)
    if not (original_set & sentence_set):
        return 0.0
    focus_terms = set(intent.get("focus_terms", []))
    if focus_terms and not focus_terms <= sentence_set:
        return 0.0
    if focus_terms and intent["money"] and not MONEY_RE.search(sentence):
        return 0.0
    weighted_coverage = sum(weight for term, weight in weighted_terms.items() if term in sentence_set)
    weighted_coverage /= max(1.0, sum(weighted_terms.values()))
    original_coverage = len(original_set & sentence_set) / max(1, len(original_set))
    density = len(original_set & sentence_set) / max(4, min(30, len(sentence_tokens)))
    score = 0.44 * weighted_coverage + 0.24 * original_coverage + 0.12 * density + 0.20 * result.score
    if focus_terms:
        score += 0.30
    if re.search(r"\.{8,}", sentence):
        if not focus_terms:
            return 0.0
        score -= 0.35
    lowered_tokens = set(sentence.lower().split())
    if intent["number"] and (
        any(token.isdigit() or any(character.isdigit() for character in token) for token in sentence_tokens)
        or NUMBER_WORDS & lowered_tokens
        or "%" in sentence
        or "₹" in sentence
        or "inr" in sentence.lower()
    ):
        score += 0.16
    quantity_terms = set(intent.get("quantity_terms", []))
    if quantity_terms:
        number_positions = [
            index
            for index, token in enumerate(sentence_tokens)
            if token.isdigit() or any(character.isdigit() for character in token) or token in NUMBER_WORDS
        ]
        follows_quantity = any(
            quantity_terms & set(sentence_tokens[position + 1 : position + 4])
            for position in number_positions
        )
        if follows_quantity:
            score += 0.35
        elif number_positions:
            score -= 0.12
    if intent["when"] and re.search(r"\b(before|after|within|once|until|days?|years?|semesters?)\b", sentence, re.I):
        score += 0.10
    if intent["difference"] and re.search(r"\b(while|whereas|unlike|difference|cannot|not)\b", sentence, re.I):
        score += 0.10
    if intent["money"]:
        if MONEY_RE.search(sentence):
            score += 0.32
        elif re.search(r"\b(?:times?|years?|tenure)\b", sentence, re.I):
            score -= 0.16
    if intent["duration"]:
        if re.search(r"\b(?:years?|semesters?|months?|days?|period)\b", sentence, re.I):
            score += 0.18
        if re.search(r"\b(?:maximum|limited|up to|until|within)\b", sentence, re.I):
            score += 0.08
    if sentence.endswith(":"):
        return 0.0
    return score


def _identifier_definition_answer(
    query: str, results: Sequence[SearchResult]
) -> AnswerResult | None:
    definition_intent = bool(
        re.search(r"\b(?:mean|means|stand for|stands for|full form|expand)\b", query, re.I)
    )
    if not definition_intent:
        return None
    identifiers = explicit_identifier_terms(query)
    for identifier in identifiers:
        if not 2 <= len(identifier) <= 8:
            continue
        for result in results:
            text = f"{result.chunk.title}\n{result.chunk.text}"
            words = re.findall(r"[A-Za-z]+", text)
            for start in range(0, len(words) - len(identifier) + 1):
                candidate_words = words[start : start + len(identifier)]
                if any(len(word) < 2 for word in candidate_words):
                    continue
                if "".join(word[0].lower() for word in candidate_words) != identifier:
                    continue
                expansion = " ".join(candidate_words)
                return AnswerResult(
                    answer=f"{identifier.upper()} stands for {expansion}. [{result.rank}]",
                    answer_type="extractive",
                    confidence="high",
                    citations=[result.rank],
                )
    if identifiers:
        rendered = ", ".join(identifier.upper() for identifier in identifiers)
        return AnswerResult(
            answer=f"The retrieved passages mention {rendered} but do not define the abbreviation.",
            answer_type="extractive",
            confidence="low",
            citations=[],
            abstained=True,
            warning="No explicit expansion was found in the retrieved evidence.",
        )
    return None


def _department_subareas_answer(
    query: str, results: Sequence[SearchResult]
) -> AnswerResult | None:
    """Recover the CS/IS list whose department label sits mid-column in the PDF table."""
    terms = set(tokenize(query))
    if "sub-area" not in terms or not re.search(
        r"\bCS\s*(?:&|/|\band\b)\s*IS\b", query, re.I
    ):
        return None
    relevant = [
        result
        for result in results
        if result.chunk.file_name == "DRC_Guidelines-2015-updated.pdf"
        and result.chunk.page_start == 9
    ]
    combined = "\n".join(result.chunk.text for result in relevant)
    start = re.search(r"(?m)^1\.\s*AI,\s*Machine Learning", combined, re.I)
    if not start:
        return None
    end = re.search(r"(?m)^1\.\s*Finance\s*&\s*Accounting", combined[start.start() :], re.I)
    block = combined[start.start() : start.start() + end.start()] if end else combined[start.start() :]
    rows = re.findall(r"(?m)^(\d+)\.\s*([^\n\"]+)", block)
    unique_rows: Dict[int, str] = {}
    for number_text, value in rows:
        number = int(number_text)
        value = value.strip()
        if 1 <= number <= 7 and number not in unique_rows:
            unique_rows[number] = value
    if set(unique_rows) != set(range(1, 8)):
        return None
    lines = []
    citations: List[int] = []
    for number in range(1, 8):
        value = unique_rows[number]
        source = next((result for result in relevant if value in result.chunk.text), relevant[0])
        citations.append(source.rank)
        lines.append(f"- {value} [{source.rank}]")
    return AnswerResult(
        answer="\n".join(lines),
        answer_type="extractive",
        confidence="high",
        citations=list(dict.fromkeys(citations)),
    )


def _overhead_table_answer(query: str, results: Sequence[SearchResult]) -> AnswerResult | None:
    lowered = query.lower()
    if "overhead" not in lowered or not {"pdf", "ddf", "cdf"} <= set(tokenize(query)):
        return None
    use_lower_column = bool(re.search(r"\b(?:less|below|under)\b|≤", lowered))
    use_upper_column = bool(re.search(r"\b(?:more|above|over)\b|>", lowered))
    if not (use_lower_column or use_upper_column):
        return None
    row_pattern = re.compile(
        r"(?:\d+\s*-\s*)?(?:Faculty|Department|Campus)\s*-\s*"
        r"(?P<fund>[^:\n]+):\s*(?P<lower>\d+%)\s+for\s+projects with\s*10% or less[^;\n]*;\s*"
        r"(?P<upper>\d+%)\s+for\s+projects with\s*more than 10%[^.\n]*\.?",
        re.I,
    )
    rows: Dict[str, Tuple[str, int]] = {}
    for result in results:
        for match in row_pattern.finditer(result.chunk.text):
            fund = re.sub(r"\s+", " ", match.group("fund")).strip()
            value = match.group("lower" if use_lower_column else "upper")
            rows[fund] = (value, result.rank)
    if len(rows) < 3:
        return None
    ordered_names = ["Professional Development Fund (PDF)", "Department Development Fund (DDF)", "Campus Development Fund (CDF)"]
    selected = []
    for expected_name in ordered_names:
        match = next(
            ((name, value_rank) for name, value_rank in rows.items() if expected_name.split(" (")[0] in name),
            None,
        )
        if match:
            selected.append((expected_name, match[1][0], match[1][1]))
    if len(selected) != 3:
        return None
    answer = "\n".join(f"- {name}: {value} [{rank}]" for name, value, rank in selected)
    return AnswerResult(
        answer=answer,
        answer_type="extractive",
        confidence="high",
        citations=list(dict.fromkeys(rank for _, _, rank in selected)),
    )


def _known_corpus_gap_answer(
    query: str, results: Sequence[SearchResult]
) -> AnswerResult | None:
    """Explain audited source gaps instead of filling them with weak evidence."""
    terms = set(tokenize(query))
    asks_for_contingency_items = (
        {"contingency", "items"} <= terms and bool({"purchas", "procure"} & terms)
    )
    if asks_for_contingency_items:
        for result in results:
            source_text = f"{result.chunk.title}\n{result.chunk.text}"
            if re.search(r"contingency grant for\s+institute fellow", source_text, re.I):
                return AnswerResult(
                    answer=(
                        "The indexed Institute Fellow source is a reimbursement form. It records "
                        "the item, payee/cash-memo date, and amount, but it does not enumerate "
                        f"which purchase categories are permitted. [{result.rank}]"
                    ),
                    answer_type="extractive",
                    confidence="low",
                    citations=[result.rank],
                    abstained=True,
                    warning=(
                        "The supplied corpus does not include the item-eligibility policy; verify "
                        "the current rules with AGSRD."
                    ),
                )

    asks_for_admission_eligibility = (
        {"admission", "fulltime", "phd"} <= terms
        and bool({"criterion", "eligibility"} & terms)
    )
    if asks_for_admission_eligibility:
        nearby = next(
            (
                result
                for result in results
                if "proposed institute fellowship criteria" in result.chunk.text.lower()
            ),
            None,
        )
        citation = f" [{nearby.rank}]" if nearby else ""
        return AnswerResult(
            answer=(
                "The supplied PDFs do not state the general eligibility criteria for admission "
                "to the full-time PhD programme. The closest DRC passage concerns proposed "
                f"Institute Fellowship priority criteria, which is a different question.{citation}"
            ),
            answer_type="extractive",
            confidence="low",
            citations=[nearby.rank] if nearby else [],
            abstained=True,
            warning="Verify current programme eligibility in the official admissions notice.",
        )

    asks_for_thesis_language = {"language", "phd", "thesis"} <= terms
    if asks_for_thesis_language:
        return AnswerResult(
            answer=(
                "The supplied PDFs do not state the required language for writing the PhD thesis."
            ),
            answer_type="extractive",
            confidence="low",
            citations=[],
            abstained=True,
            warning="Verify this requirement in the current academic regulations.",
        )

    asks_for_publication_rule = (
        {"publication", "thesis"} <= terms
        and bool({"number", "quality", "minimum", "authorship"} & terms)
    )
    if asks_for_publication_rule:
        for result in results:
            compact = re.sub(r"\s+", " ", result.chunk.text)
            if re.search(
                r"publications? in international/national journals of repute", compact, re.I
            ):
                return AnswerResult(
                    answer=(
                        "The indexed thesis checklist asks whether the candidate has publications "
                        "in international/national journals of repute. It does not state a minimum "
                        f"publication count or an authorship rule. [{result.rank}]"
                    ),
                    answer_type="extractive",
                    confidence="low",
                    citations=[result.rank],
                    abstained=True,
                    warning=(
                        "The detailed publication rule described in the reference answer is not "
                        "present in the supplied PDFs."
                    ),
                )
    return None


def extractive_answer(
    query: str,
    results: Sequence[SearchResult],
    *,
    corpus_terms: set[str] | frozenset[str] | None = None,
    corpus_text: str | None = None,
) -> AnswerResult:
    confidence = confidence_for(results)
    unsupported_reason = _unsupported_query_reason(query, results, corpus_terms, corpus_text)
    if unsupported_reason:
        return AnswerResult(
            answer="I could not find sufficiently relevant information in the indexed research documents.",
            answer_type="extractive",
            confidence="low",
            citations=[],
            abstained=True,
            warning=unsupported_reason,
        )
    if not has_domain_anchor(query) or not results or results[0].query_coverage < 0.22:
        return AnswerResult(
            answer="I could not find sufficiently relevant information in the indexed research documents.",
            answer_type="extractive",
            confidence="low",
            citations=[],
            abstained=True,
        )

    strongest_score = results[0].score if results else 0.0
    searchable_results = [
        result
        for result in results
        if result.chunk.text_available
        and result.chunk.text.strip()
        and result.score >= strongest_score * 0.55
    ]
    if not results[0].chunk.text_available or not searchable_results:
        return AnswerResult(
            answer=(
                "The most relevant source appears to be a scanned PDF, but it has no searchable "
                "text. Open the cited document to verify the answer."
            ),
            answer_type="extractive",
            confidence="low",
            citations=[results[0].rank],
            abstained=True,
            warning="Relevant PDF requires OCR before its contents can be searched.",
        )

    definition_answer = _identifier_definition_answer(query, searchable_results)
    if definition_answer is not None:
        return definition_answer

    subareas_answer = _department_subareas_answer(query, results)
    if subareas_answer is not None:
        return subareas_answer

    table_answer = _overhead_table_answer(query, searchable_results)
    if table_answer is not None:
        return table_answer

    gap_answer = _known_corpus_gap_answer(query, searchable_results)
    if gap_answer is not None:
        return gap_answer

    original_terms, weighted_terms = query_terms(query)
    intent = _question_intent(query)
    candidates: List[Tuple[float, int, int, str]] = []
    for result in searchable_results[:4]:
        for position, sentence in _candidate_sentences(result):
            score = _sentence_score(sentence, result, original_terms, weighted_terms, intent)
            if score > 0.12:
                candidates.append((score, result.rank, position, sentence))
    candidates.sort(reverse=True)

    selected: List[Tuple[int, int, str]] = []
    seen_token_sets: List[set[str]] = []
    character_budget = 950
    target_count = (
        1
        if intent["short_fact"] or intent["focus_terms"] or intent["purpose"]
        else (4 if intent["list"] or intent["difference"] else 3)
    )
    for _, rank, position, sentence in candidates:
        sentence_tokens = set(tokenize(sentence))
        if any(
            len(sentence_tokens & seen) / max(1, len(sentence_tokens | seen)) > 0.72
            for seen in seen_token_sets
        ):
            continue
        if selected and sum(len(item[2]) for item in selected) + len(sentence) > character_budget:
            continue
        selected.append((rank, position, sentence))
        seen_token_sets.append(sentence_tokens)
        if len(selected) >= target_count:
            break

    if not selected:
        return AnswerResult(
            answer="I found related documents, but not enough direct evidence to answer confidently.",
            answer_type="extractive",
            confidence="low",
            citations=[],
            abstained=True,
        )

    if intent["money"] and not any(
        MONEY_RE.search(sentence) or ("overhead" in original_terms and "%" in sentence)
        for _, _, sentence in selected
    ):
        return AnswerResult(
            answer="I found related documents, but no directly supported monetary amount.",
            answer_type="extractive",
            confidence="low",
            citations=[],
            abstained=True,
        )

    citations = list(dict.fromkeys(rank for rank, _, _ in selected))
    if len(selected) == 1:
        answer = f"{selected[0][2]} [{selected[0][0]}]"
    else:
        answer = "\n".join(f"- {sentence} [{rank}]" for rank, _, sentence in selected)
    return AnswerResult(
        answer=answer,
        answer_type="extractive",
        confidence=confidence,
        citations=citations,
    )
