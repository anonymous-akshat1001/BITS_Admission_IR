# Evaluation Guide

The project compares two local, deterministic systems:

1. **Legacy-style TF-IDF proxy** — whole PDFs are lowercased, converted to ASCII, and collapsed to one line; text is split into fixed 600-character chunks with 150-character overlap; unigram TF-IDF cosine ranks chunks; the highest-ranked chunk is returned as the answer.
2. **Improved local SearchService** — page-aware extraction, structure-preserving chunks, explainable hybrid lexical ranking, result diversity, focused extractive answers, citations, and rule- and relevance-based abstention.

> The legacy score is a reproducible local proxy, not a recovered historical result. The original repository did not contain its Qdrant index, credentials, or saved evaluation results. TF-IDF replaces those unavailable services while retaining the original pipeline's most visible preprocessing, chunking, and top-chunk-answer behavior.

## Data

- `data/evaluation/queries.csv`: 44 stable query IDs, questions, and reference answers.
- `data/evaluation/qrels.csv`: query IDs, categories, and one or more relevant PDF filenames.
- `data/evaluation/source_gaps.csv`: four query IDs whose complete reference answer is absent from the supplied corpus, with an audit reason.
- `data/evaluation/unanswerable.csv`: 18 deliberately unsupported questions: four simple no-domain-anchor cases, eight hard negatives, and six adversarial paraphrases that reuse corpus terms.

Query IDs are stored in the query CSV and validated against both qrels and source-gap labels, so reordering rows cannot silently change judgments. Relevant filenames in one qrels cell are separated with semicolons.

Manual source inspection found four reference answers that are not fully supported by the bundled PDFs: Q06's form does not enumerate permitted contingency purchases, Q17 lacks a thesis-language rule, Q36 lacks the general programme-admission criteria, and Q44 lacks the stated publication-count/authorship policy. Their qrels identify the closest topical documents for retrieval diagnostics, but those hits are not evidence-retrieval successes. A safe system should report the gap. Aggregate answer-overlap metrics therefore use only the 40 corpus-supported questions, while source-gap abstention is reported separately.

## Run

From the repository root:

```bash
python -m backend.scripts.evaluate
```

The only non-standard runtime dependency is PyMuPDF. Two checksum-verified, manually reviewed OCR sidecars are bundled for the image-only policy PDFs, so a normal evaluation does not need Tesseract. The following option is only a fallback for newly added image-only PDFs when Tesseract is installed:

```bash
python -m backend.scripts.evaluate --enable-ocr
```

Use `python -m backend.scripts.evaluate --help` to override corpus, data, or output paths. The normal output directory is `artifacts/evaluation/`.

## Outputs

- `per_query.csv`: query text, source judgments, ranked document names, answer, abstention/error state, latency, and every per-query metric for both systems.
- `summary.json`: machine-readable input, implementation, OCR-sidecar, and effective-corpus hashes plus runtime metadata, diagnostics, and category scores. These hashes make a dirty working-tree evaluation identifiable even when its Git commit alone does not.
- `summary.md`: presentation-ready overall and per-category comparison tables.

Failures remain in metric denominators with zero quality scores. This prevents a system from appearing better by failing on difficult questions. Failure rate is also reported explicitly.

## Metric definitions

- **Document Hit@1/3/5:** 1 when at least one relevant PDF occurs by the cutoff, otherwise 0; averaged across queries.
- **MRR@10:** reciprocal rank of the first relevant PDF, capped at rank 10.
- **nDCG@5:** binary, document-level discounted cumulative gain normalized by the ideal ranking. It gives partial credit for ranking relevant documents lower and supports queries with multiple relevant PDFs.
- **Supported answer token-F1:** bag-of-token precision/recall F1 between the extractive answer and reference answer, averaged only over the 40 corpus-supported questions.
- **Supported ROUGE-L:** token-level longest-common-subsequence F1 over the same supported subset.
- **Post-warm-up p50/p95 latency:** wall-clock percentiles for successful measured calls across supported, source-gap, and OOD queries after initialization and one unmeasured warm-up query. Index construction is excluded.
- **Failure rate:** failed answerable and unanswerable calls divided by all attempted calls.
- **Source-gap abstention accuracy:** fraction of the four audited corpus-gap questions for which the system avoids inventing the missing policy detail.
- **Unanswerable abstention accuracy:** fraction of unsupported queries for which the system explicitly abstains, also split into no-domain-anchor and hard-negative categories.

Numeric citation markers such as `[1]` are removed before answer metrics. For document metrics, repeated filenames are collapsed at their first occurrence among the ten returned chunks.

## Interpreting results

Retrieval metrics judge document-level topical relevance, not whether a passage contains every fact in a reference answer. This distinction matters for the four labeled source gaps. Token-F1 and ROUGE-L measure wording overlap, not factual correctness; concise correct answers can score lower than long copied passages. Inspect the per-query CSV for confusable question pairs, scanned documents, and category-specific regressions before drawing conclusions from averages.

The current qrels are binary, document-level, and were used while tuning the improved ranking and abstention rules. In particular, the perfect gap/OOD abstention score includes explicit scope and known-source-gap checks. The results are therefore an in-sample diagnostic, not held-out evidence of generalization. A realistic future extension is an independently authored test set with manual page- or passage-level judgments, especially for long DRC and GCIR documents; graded nDCG should only be introduced after graded labels exist.

## Tests

Run all backend tests with:

```bash
python -m pytest -q
```
