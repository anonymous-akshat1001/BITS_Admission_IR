"""Compare the reproducible legacy proxy with the improved local search service."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set


if __package__ in {None, ""}:  # Support both `python -m ...` and direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.ir_system.baseline import LegacyTfidfBaseline
from backend.ir_system.config import PROJECT_ROOT, Settings, settings
from backend.ir_system.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
    rouge_l_f1,
    token_f1,
    unique_documents,
)
from backend.ir_system.service import SearchService


logger = logging.getLogger(__name__)
RETRIEVAL_DEPTH = 10
BASELINE_LIMITATION = (
    "The legacy result is a reproducible local proxy, not a historical measurement. "
    "The original repository did not include its Qdrant index, credentials, or saved results; "
    "the proxy retains its whole-document lossy cleanup, 600-character chunks, and top-chunk "
    "answer, while using transparent unigram TF-IDF in place of unavailable vector services."
)


@dataclass(frozen=True)
class AnswerableCase:
    query_id: str
    category: str
    question: str
    reference_answer: str
    relevant_documents: Set[str]
    corpus_supported: bool


@dataclass(frozen=True)
class UnanswerableCase:
    query_id: str
    category: str
    question: str


@dataclass(frozen=True)
class RunOutput:
    answer: str
    retrieved_documents: List[str]
    abstained: bool


class LegacyRunner:
    name = "legacy_proxy"
    display_name = "Legacy-style TF-IDF proxy"

    def __init__(self, corpus_dir: Path):
        started = time.perf_counter()
        self.index = LegacyTfidfBaseline.from_corpus(corpus_dir)
        self.initialization_ms = (time.perf_counter() - started) * 1000.0

    def run(self, query: str) -> RunOutput:
        results = self.index.search(query, top_k=RETRIEVAL_DEPTH)
        return RunOutput(
            answer=self.index.answer(results),
            retrieved_documents=[result.chunk.file_name for result in results],
            # The original nearest-result behavior had no relevance-based refusal rule.
            abstained=False,
        )


class ImprovedRunner:
    name = "improved"
    display_name = "Improved local SearchService"

    def __init__(self, app_settings: Settings):
        started = time.perf_counter()
        self.service = SearchService(app_settings)
        self.service.ensure_ready()
        self.initialization_ms = (time.perf_counter() - started) * 1000.0

    def run(self, query: str) -> RunOutput:
        result = self.service.query(
            query,
            top_k=RETRIEVAL_DEPTH,
            answer_mode="extractive",
        )
        return RunOutput(
            answer=result.answer.answer,
            retrieved_documents=[item.chunk.file_name for item in result.results],
            abstained=result.answer.abstained,
        )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_source_gap_ids(path: Path) -> Set[str]:
    rows = _read_csv(path)
    gap_ids: Set[str] = set()
    for row in rows:
        query_id = (row.get("query_id") or "").strip()
        reason = (row.get("reason") or "").strip()
        if not query_id or not reason:
            raise ValueError(f"Invalid source-gap row: {row}")
        if query_id in gap_ids:
            raise ValueError(f"Duplicate source-gap query_id: {query_id}")
        gap_ids.add(query_id)
    return gap_ids


def load_answerable_cases(
    queries_path: Path, qrels_path: Path, source_gaps_path: Path
) -> List[AnswerableCase]:
    query_rows = _read_csv(queries_path)
    qrel_rows = _read_csv(qrels_path)
    source_gap_ids = load_source_gap_ids(source_gaps_path)
    qrels: Dict[str, Dict[str, object]] = {}
    for row in qrel_rows:
        query_id = (row.get("query_id") or "").strip()
        category = (row.get("category") or "").strip()
        documents = {
            document.strip()
            for document in (row.get("relevant_documents") or "").split(";")
            if document.strip()
        }
        if not query_id or not category or not documents:
            raise ValueError(f"Invalid qrels row: {row}")
        if query_id in qrels:
            raise ValueError(f"Duplicate qrels query_id: {query_id}")
        qrels[query_id] = {"category": category, "documents": documents}

    cases: List[AnswerableCase] = []
    seen_ids: Set[str] = set()
    for row in query_rows:
        query_id = (row.get("query_id") or "").strip()
        question = (row.get("Question") or row.get("question") or "").strip()
        answer = (row.get("Answer") or row.get("answer") or "").strip()
        if not query_id or not question or not answer:
            raise ValueError(f"Query {query_id} must contain Question and Answer")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id in queries: {query_id}")
        if query_id not in qrels:
            raise ValueError(f"No qrels entry for query {query_id}")
        seen_ids.add(query_id)
        qrel = qrels[query_id]
        cases.append(
            AnswerableCase(
                query_id=query_id,
                category=str(qrel["category"]),
                question=question,
                reference_answer=answer,
                relevant_documents=set(qrel["documents"]),
                corpus_supported=query_id not in source_gap_ids,
            )
        )

    unused_qrels = set(qrels) - seen_ids
    if unused_qrels:
        raise ValueError(f"Qrels contain unknown query IDs: {', '.join(sorted(unused_qrels))}")
    unknown_gaps = source_gap_ids - seen_ids
    if unknown_gaps:
        raise ValueError(
            f"Source gaps contain unknown query IDs: {', '.join(sorted(unknown_gaps))}"
        )
    return cases


def load_unanswerable_cases(path: Path) -> List[UnanswerableCase]:
    cases: List[UnanswerableCase] = []
    seen_ids: Set[str] = set()
    for row in _read_csv(path):
        query_id = (row.get("query_id") or "").strip()
        category = (row.get("category") or "").strip()
        question = (row.get("Question") or row.get("question") or "").strip()
        if not query_id or not category or not question:
            raise ValueError(f"Invalid unanswerable row: {row}")
        if query_id in seen_ids:
            raise ValueError(f"Duplicate unanswerable query_id: {query_id}")
        seen_ids.add(query_id)
        cases.append(UnanswerableCase(query_id=query_id, category=category, question=question))
    return cases


def _base_record(runner, dataset: str, query_id: str, category: str, question: str) -> Dict[str, object]:
    return {
        "system": runner.name,
        "system_name": runner.display_name,
        "dataset": dataset,
        "corpus_supported": None,
        "query_id": query_id,
        "category": category,
        "question": question,
        "reference_answer": "",
        "generated_answer": "",
        "abstained": False,
        "success": False,
        "error": "",
        "latency_ms": 0.0,
        "relevant_documents": [],
        "retrieved_documents": [],
        "top_document": "",
        "hit_at_1": None,
        "hit_at_3": None,
        "hit_at_5": None,
        "mrr_at_10": None,
        "ndcg_at_5": None,
        "answer_token_f1": None,
        "rouge_l_f1": None,
        "abstention_correct": None,
    }


def evaluate_runner(
    runner,
    answerable_cases: Sequence[AnswerableCase],
    unanswerable_cases: Sequence[UnanswerableCase],
) -> List[Dict[str, object]]:
    """Evaluate one initialized runner; a per-query exception becomes a scored failure."""

    if answerable_cases:
        runner.run(answerable_cases[0].question)  # Warm caches without recording this call.

    records: List[Dict[str, object]] = []
    for case in answerable_cases:
        dataset = "answerable" if case.corpus_supported else "source_gap"
        record = _base_record(
            runner, dataset, case.query_id, case.category, case.question
        )
        record["corpus_supported"] = case.corpus_supported
        record["reference_answer"] = case.reference_answer
        record["relevant_documents"] = sorted(case.relevant_documents)
        started = time.perf_counter()
        try:
            output = runner.run(case.question)
            record["latency_ms"] = (time.perf_counter() - started) * 1000.0
            ranked_documents = unique_documents(output.retrieved_documents)
            record.update(
                {
                    "generated_answer": output.answer,
                    "abstained": output.abstained,
                    "success": True,
                    "retrieved_documents": ranked_documents,
                    "top_document": ranked_documents[0] if ranked_documents else "",
                    "hit_at_1": hit_at_k(ranked_documents, case.relevant_documents, 1),
                    "hit_at_3": hit_at_k(ranked_documents, case.relevant_documents, 3),
                    "hit_at_5": hit_at_k(ranked_documents, case.relevant_documents, 5),
                    "mrr_at_10": reciprocal_rank(
                        ranked_documents, case.relevant_documents, cutoff=10
                    ),
                    "ndcg_at_5": ndcg_at_k(ranked_documents, case.relevant_documents, 5),
                    "answer_token_f1": token_f1(case.reference_answer, output.answer),
                    "rouge_l_f1": rouge_l_f1(case.reference_answer, output.answer),
                }
            )
        except Exception as exc:  # Continue so failure rate and missing scores stay visible.
            record["latency_ms"] = (time.perf_counter() - started) * 1000.0
            record["error"] = f"{type(exc).__name__}: {exc}"
            for metric in (
                "hit_at_1",
                "hit_at_3",
                "hit_at_5",
                "mrr_at_10",
                "ndcg_at_5",
                "answer_token_f1",
                "rouge_l_f1",
            ):
                record[metric] = 0.0
            logger.exception("%s failed on %s", runner.name, case.query_id)
        records.append(record)

    for case in unanswerable_cases:
        record = _base_record(
            runner, "unanswerable", case.query_id, case.category, case.question
        )
        started = time.perf_counter()
        try:
            output = runner.run(case.question)
            record["latency_ms"] = (time.perf_counter() - started) * 1000.0
            ranked_documents = unique_documents(output.retrieved_documents)
            record.update(
                {
                    "generated_answer": output.answer,
                    "abstained": output.abstained,
                    "success": True,
                    "retrieved_documents": ranked_documents,
                    "top_document": ranked_documents[0] if ranked_documents else "",
                    "abstention_correct": float(output.abstained),
                }
            )
        except Exception as exc:
            record["latency_ms"] = (time.perf_counter() - started) * 1000.0
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["abstention_correct"] = 0.0
            logger.exception("%s failed on %s", runner.name, case.query_id)
        records.append(record)
    return records


RETRIEVAL_METRICS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr_at_10",
    "ndcg_at_5",
)
ANSWER_METRICS = (
    "answer_token_f1",
    "rouge_l_f1",
)


def _quality_summary(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    retrieval = {
        metric: mean([float(record[metric] or 0.0) for record in records])
        for metric in RETRIEVAL_METRICS
    }
    supported = [record for record in records if record.get("corpus_supported") is True]
    answers = {
        metric: mean([float(record[metric] or 0.0) for record in supported])
        for metric in ANSWER_METRICS
    }
    return {**retrieval, **answers}


def summarize_system(
    runner,
    records: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    answerable = [record for record in records if record["dataset"] == "answerable"]
    source_gaps = [record for record in records if record["dataset"] == "source_gap"]
    reference_cases = answerable + source_gaps
    unanswerable = [record for record in records if record["dataset"] == "unanswerable"]
    successful_latencies = [
        float(record["latency_ms"])
        for record in records
        if record["success"]
    ]
    failures = sum(not bool(record["success"]) for record in records)
    overall: Dict[str, object] = {
        "answerable_query_count": len(answerable),
        "source_gap_query_count": len(source_gaps),
        "unanswerable_query_count": len(unanswerable),
        "evaluated_query_count": len(records),
        **_quality_summary(reference_cases),
        "latency_p50_ms": percentile(successful_latencies, 0.50),
        "latency_p95_ms": percentile(successful_latencies, 0.95),
        "failure_count": failures,
        "failure_rate": failures / len(records) if records else 0.0,
        "unanswerable_abstention_accuracy": mean(
            [float(record["abstention_correct"] or 0.0) for record in unanswerable]
        ),
        "answerable_abstention_rate": mean(
            [float(bool(record["abstained"])) for record in answerable]
        ),
        "source_gap_abstention_accuracy": mean(
            [float(bool(record["abstained"])) for record in source_gaps]
        ),
    }

    categories: Dict[str, Dict[str, object]] = {}
    for category in sorted({str(record["category"]) for record in reference_cases}):
        subset = [record for record in reference_cases if record["category"] == category]
        category_failures = sum(not bool(record["success"]) for record in subset)
        category_latencies = [
            float(record["latency_ms"]) for record in subset if record["success"]
        ]
        categories[category] = {
            "query_count": len(subset),
            "supported_answer_query_count": sum(
                record.get("corpus_supported") is True for record in subset
            ),
            "source_gap_query_count": sum(
                record.get("corpus_supported") is False for record in subset
            ),
            **_quality_summary(subset),
            "latency_p50_ms": percentile(category_latencies, 0.50),
            "latency_p95_ms": percentile(category_latencies, 0.95),
            "failure_rate": category_failures / len(subset) if subset else 0.0,
        }
    unanswerable_categories: Dict[str, Dict[str, object]] = {}
    for category in sorted({str(record["category"]) for record in unanswerable}):
        subset = [record for record in unanswerable if record["category"] == category]
        unanswerable_categories[category] = {
            "query_count": len(subset),
            "abstention_accuracy": mean(
                [float(record["abstention_correct"] or 0.0) for record in subset]
            ),
        }
    return {
        "display_name": runner.display_name,
        "initialization_ms": runner.initialization_ms,
        "overall": overall,
        "categories": categories,
        "unanswerable_categories": unanswerable_categories,
    }


CSV_FIELDS = (
    "system",
    "system_name",
    "dataset",
    "corpus_supported",
    "query_id",
    "category",
    "question",
    "reference_answer",
    "generated_answer",
    "abstained",
    "success",
    "error",
    "latency_ms",
    "relevant_documents",
    "retrieved_documents",
    "top_document",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "mrr_at_10",
    "ndcg_at_5",
    "answer_token_f1",
    "rouge_l_f1",
    "abstention_correct",
)


def write_per_query_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            serialized = dict(record)
            serialized["relevant_documents"] = ";".join(record["relevant_documents"])
            serialized["retrieved_documents"] = ";".join(record["retrieved_documents"])
            writer.writerow(serialized)


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _score(value: object) -> str:
    return f"{float(value):.3f}"


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative path in saved, shareable reports."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: _portable_path(item).lower()):
        digest.update(_portable_path(path).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _file_hashes(paths: Sequence[Path]) -> Dict[str, str]:
    return {
        _portable_path(path): _sha256(path)
        for path in sorted(paths, key=lambda item: _portable_path(item).lower())
    }


def _git_state() -> Dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "working_tree_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "working_tree_dirty": None}


def _run_environment(app_settings: Settings, corpus_dir: Path) -> Dict[str, object]:
    try:
        import fitz

        pymupdf_version = fitz.VersionBind
    except (ImportError, AttributeError):
        pymupdf_version = None
    pdf_files = list(corpus_dir.glob("*.pdf"))
    ocr_sidecars = list(app_settings.ocr_dir.glob("*.json"))
    implementation_files = [
        *list((PROJECT_ROOT / "backend" / "ir_system").glob("*.py")),
        PROJECT_ROOT / "backend" / "scripts" / "evaluate.py",
        PROJECT_ROOT / "backend" / "requirements.txt",
        PROJECT_ROOT / "backend" / "requirements-dev.txt",
    ]
    implementation_files = [path for path in implementation_files if path.is_file()]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "pymupdf": pymupdf_version,
        "pdf_corpus_sha256": _files_fingerprint(pdf_files),
        "effective_corpus_sha256": _files_fingerprint([*pdf_files, *ocr_sidecars]),
        "ocr_sidecar_sha256": _file_hashes(ocr_sidecars),
        "implementation_sha256": _file_hashes(implementation_files),
        "settings": {
            "chunk_size": app_settings.chunk_size,
            "chunk_overlap": app_settings.chunk_overlap,
            "max_chunks_per_document": app_settings.max_chunks_per_document,
            "ocr_enabled": app_settings.ocr_enabled,
            "answer_mode": "extractive",
            "retrieval_depth_chunks": RETRIEVAL_DEPTH,
        },
        "git": _git_state(),
    }


def build_markdown(summary: Dict[str, object]) -> str:
    systems = summary["systems"]
    lines = [
        "# Information Retrieval Evaluation",
        "",
        f"> **Baseline limitation:** {BASELINE_LIMITATION}",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        "## Overall comparison",
        "",
        "| System | Hit@1* | Hit@3* | Hit@5* | MRR@10* | nDCG@5* | Supported Token-F1 | Supported ROUGE-L | p50 ms | p95 ms | Failure | Supported abstention | Gap abstention | OOD abstention |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in systems.values():
        overall = system["overall"]
        lines.append(
            "| {name} | {h1} | {h3} | {h5} | {mrr} | {ndcg} | {f1} | {rouge} | "
            "{p50:.2f} | {p95:.2f} | {failure} | {answerable_abstention} | "
            "{gap_abstention} | {abstention} |".format(
                name=system["display_name"],
                h1=_percent(overall["hit_at_1"]),
                h3=_percent(overall["hit_at_3"]),
                h5=_percent(overall["hit_at_5"]),
                mrr=_score(overall["mrr_at_10"]),
                ndcg=_score(overall["ndcg_at_5"]),
                f1=_score(overall["answer_token_f1"]),
                rouge=_score(overall["rouge_l_f1"]),
                p50=float(overall["latency_p50_ms"]),
                p95=float(overall["latency_p95_ms"]),
                failure=_percent(overall["failure_rate"]),
                answerable_abstention=_percent(overall["answerable_abstention_rate"]),
                gap_abstention=_percent(overall["source_gap_abstention_accuracy"]),
                abstention=_percent(overall["unanswerable_abstention_accuracy"]),
            )
        )

    lines.extend(
        [
            "",
            "\\* Retrieval qrels judge the best topical document for all 44 reference queries. For the four source-gap queries, that document does not contain the complete reference answer; these are not evidence-retrieval successes.",
        ]
    )

    lines.extend(
        [
            "",
            "## Results by reference-query category",
            "",
            "| System | Category | Queries | Supported | Gaps | Hit@1 | Hit@3 | MRR@10 | nDCG@5 | Supported Token-F1 | Supported ROUGE-L | p50 ms | Failure |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system in systems.values():
        for category, metrics in system["categories"].items():
            lines.append(
                "| {name} | {category} | {count} | {supported} | {gaps} | {h1} | {h3} | {mrr} | {ndcg} | {f1} | {rouge} | {p50:.2f} | {failure} |".format(
                    name=system["display_name"],
                    category=category,
                    count=metrics["query_count"],
                    supported=metrics["supported_answer_query_count"],
                    gaps=metrics["source_gap_query_count"],
                    h1=_percent(metrics["hit_at_1"]),
                    h3=_percent(metrics["hit_at_3"]),
                    mrr=_score(metrics["mrr_at_10"]),
                    ndcg=_score(metrics["ndcg_at_5"]),
                    f1=(
                        _score(metrics["answer_token_f1"])
                        if metrics["supported_answer_query_count"]
                        else "—"
                    ),
                    rouge=(
                        _score(metrics["rouge_l_f1"])
                        if metrics["supported_answer_query_count"]
                        else "—"
                    ),
                    p50=float(metrics["latency_p50_ms"]),
                    failure=_percent(metrics["failure_rate"]),
                )
            )

    lines.extend(
        [
            "",
            "## Unanswerable-query breakdown",
            "",
            "| System | Category | Queries | Abstention accuracy |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for system in systems.values():
        for category, metrics in system["unanswerable_categories"].items():
            lines.append(
                "| {name} | {category} | {count} | {accuracy} |".format(
                    name=system["display_name"],
                    category=category,
                    count=metrics["query_count"],
                    accuracy=_percent(metrics["abstention_accuracy"]),
                )
            )

    lines.extend(
        [
            "",
            "## Method",
            "",
            f"- {summary['answerable_query_count']} corpus-supported questions and {summary['source_gap_query_count']} audited source-gap questions use manually assigned, binary document-level qrels.",
            f"- {summary['unanswerable_query_count']} out-of-scope questions measure abstention accuracy.",
            "- Query IDs are stored in the query CSV and validated across qrels and source-gap labels.",
            "- Each system is initialized and receives one unmeasured warm-up query before timing.",
            "- Up to 10 ranked chunks are requested; repeated filenames are collapsed at their first occurrence for document metrics.",
            "- Failed queries remain in quality denominators with zero scores and are also reported as failures.",
            "- Token-F1 and ROUGE-L are calculated only on corpus-supported questions and ignore numeric citation markers such as `[1]`.",
            "- Q06, Q17, Q36, and Q44 are source gaps; their answer-overlap values remain in the detailed CSV but are excluded from aggregate answer quality.",
            "- Latency percentiles include successful measured calls across supported, source-gap, and OOD queries after one unmeasured warm-up; initialization is excluded.",
            "- This is an in-sample diagnostic set used during development, not a held-out estimate of generalization.",
            "",
            "The detailed CSV should be inspected for category-specific wins, errors, abstentions, and confusable questions; aggregate scores alone do not establish factual correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    *,
    corpus_dir: Path,
    queries_path: Path,
    qrels_path: Path,
    source_gaps_path: Path,
    unanswerable_path: Path,
    output_dir: Path,
    enable_ocr: bool = False,
) -> Dict[str, Path]:
    answerable_cases = load_answerable_cases(queries_path, qrels_path, source_gaps_path)
    unanswerable_cases = load_unanswerable_cases(unanswerable_path)
    corpus_files = {path.name for path in corpus_dir.glob("*.pdf")}
    judged_files = {
        file_name for case in answerable_cases for file_name in case.relevant_documents
    }
    missing_judged_files = sorted(judged_files - corpus_files)
    if missing_judged_files:
        raise ValueError(
            "Qrels reference PDFs absent from the corpus: "
            + ", ".join(missing_judged_files)
        )
    app_settings = Settings(
        corpus_dir=corpus_dir,
        evaluation_dataset=queries_path,
        unanswerable_dataset=unanswerable_path,
        artifacts_dir=output_dir,
        ocr_enabled=enable_ocr,
        gemini_api_key=None,
    )

    runners = [LegacyRunner(corpus_dir), ImprovedRunner(app_settings)]
    records: List[Dict[str, object]] = []
    system_summaries: Dict[str, object] = {}
    for runner in runners:
        runner_records = evaluate_runner(runner, answerable_cases, unanswerable_cases)
        records.extend(runner_records)
        system_summaries[runner.name] = summarize_system(runner, runner_records)

    improved_summary = runners[1].service.summary
    summary: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_limitation": BASELINE_LIMITATION,
        "answerable_query_count": sum(case.corpus_supported for case in answerable_cases),
        "source_gap_query_count": sum(not case.corpus_supported for case in answerable_cases),
        "unanswerable_query_count": len(unanswerable_cases),
        "retrieval_depth_chunks": RETRIEVAL_DEPTH,
        "inputs": {
            "corpus_dir": _portable_path(corpus_dir),
            "queries": _portable_path(queries_path),
            "qrels": _portable_path(qrels_path),
            "source_gaps": _portable_path(source_gaps_path),
            "unanswerable": _portable_path(unanswerable_path),
        },
        "input_sha256": {
            "queries": _sha256(queries_path),
            "qrels": _sha256(qrels_path),
            "source_gaps": _sha256(source_gaps_path),
            "unanswerable": _sha256(unanswerable_path),
        },
        "run_environment": _run_environment(app_settings, corpus_dir),
        "corpus": improved_summary.to_dict() if improved_summary else None,
        "metric_notes": {
            "document_ranking": "First occurrence of each filename among up to 10 ranked chunks.",
            "relevance": "Binary document-level judgments.",
            "latency": (
                "Successful measured calls across supported, source-gap, and OOD queries after "
                "one unmeasured warm-up; initialization excluded."
            ),
            "failures": "Retained as zero quality scores and included in failure rate.",
        },
        "systems": system_summaries,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / "per_query.csv",
        "json": output_dir / "summary.json",
        "markdown": output_dir / "summary.md",
    }
    write_per_query_csv(paths["csv"], records)
    paths["json"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["markdown"].write_text(build_markdown(summary), encoding="utf-8")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the reproducible legacy proxy with the improved local IR service."
    )
    parser.add_argument("--corpus", type=Path, default=settings.corpus_dir)
    parser.add_argument("--queries", type=Path, default=settings.evaluation_dataset)
    parser.add_argument(
        "--qrels", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "qrels.csv"
    )
    parser.add_argument(
        "--source-gaps",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "source_gaps.csv",
    )
    parser.add_argument("--unanswerable", type=Path, default=settings.unanswerable_dataset)
    parser.add_argument(
        "--output-dir", type=Path, default=settings.artifacts_dir / "evaluation"
    )
    parser.add_argument(
        "--enable-ocr",
        action="store_true",
        help="OCR missing image-only pages at runtime (reviewed sidecars are already bundled).",
    )
    parser.add_argument("--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    paths = run_evaluation(
        corpus_dir=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
        source_gaps_path=args.source_gaps,
        unanswerable_path=args.unanswerable,
        output_dir=args.output_dir,
        enable_ocr=args.enable_ocr,
    )
    print("Evaluation complete.")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
