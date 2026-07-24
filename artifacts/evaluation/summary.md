# Information Retrieval Evaluation

> **Baseline limitation:** The legacy result is a reproducible local proxy, not a historical measurement. The original repository did not include its Qdrant index, credentials, or saved results; the proxy retains its whole-document lossy cleanup, 600-character chunks, and top-chunk answer, while using transparent unigram TF-IDF in place of unavailable vector services.

Generated: 2026-07-22T22:35:15.715638+00:00

## Overall comparison

| System | Hit@1* | Hit@3* | Hit@5* | MRR@10* | nDCG@5* | Supported Token-F1 | Supported ROUGE-L | p50 ms | p95 ms | Failure | Supported abstention | Gap abstention | OOD abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy-style TF-IDF proxy | 79.5% | 90.9% | 93.2% | 0.854 | 0.853 | 0.215 | 0.181 | 0.21 | 0.26 | 0.0% | 0.0% | 0.0% | 0.0% |
| Improved local SearchService | 100.0% | 100.0% | 100.0% | 1.000 | 0.982 | 0.381 | 0.320 | 9.96 | 16.29 | 0.0% | 0.0% | 100.0% | 100.0% |

\* Retrieval qrels judge the best topical document for all 44 reference queries. For the four source-gap queries, that document does not contain the complete reference answer; these are not evidence-retrieval successes.

## Results by reference-query category

| System | Category | Queries | Supported | Gaps | Hit@1 | Hit@3 | MRR@10 | nDCG@5 | Supported Token-F1 | Supported ROUGE-L | p50 ms | Failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy-style TF-IDF proxy | drc | 11 | 10 | 1 | 90.9% | 100.0% | 0.955 | 0.959 | 0.272 | 0.236 | 0.22 | 0.0% |
| Legacy-style TF-IDF proxy | fellowship | 1 | 0 | 1 | 100.0% | 100.0% | 1.000 | 1.000 | — | — | 0.26 | 0.0% |
| Legacy-style TF-IDF proxy | gcir | 18 | 18 | 0 | 94.4% | 100.0% | 0.972 | 0.979 | 0.142 | 0.121 | 0.21 | 0.0% |
| Legacy-style TF-IDF proxy | programme | 9 | 7 | 2 | 55.6% | 77.8% | 0.676 | 0.662 | 0.285 | 0.216 | 0.23 | 0.0% |
| Legacy-style TF-IDF proxy | proposal | 1 | 1 | 0 | 0.0% | 100.0% | 0.500 | 0.387 | 0.094 | 0.079 | 0.25 | 0.0% |
| Legacy-style TF-IDF proxy | travel | 4 | 4 | 0 | 50.0% | 50.0% | 0.500 | 0.500 | 0.307 | 0.284 | 0.23 | 0.0% |
| Improved local SearchService | drc | 11 | 10 | 1 | 100.0% | 100.0% | 1.000 | 0.965 | 0.529 | 0.470 | 10.41 | 0.0% |
| Improved local SearchService | fellowship | 1 | 0 | 1 | 100.0% | 100.0% | 1.000 | 1.000 | — | — | 9.56 | 0.0% |
| Improved local SearchService | gcir | 18 | 18 | 0 | 100.0% | 100.0% | 1.000 | 1.000 | 0.279 | 0.225 | 12.09 | 0.0% |
| Improved local SearchService | programme | 9 | 7 | 2 | 100.0% | 100.0% | 1.000 | 0.957 | 0.367 | 0.278 | 9.99 | 0.0% |
| Improved local SearchService | proposal | 1 | 1 | 0 | 100.0% | 100.0% | 1.000 | 1.000 | 0.211 | 0.126 | 16.12 | 0.0% |
| Improved local SearchService | travel | 4 | 4 | 0 | 100.0% | 100.0% | 1.000 | 1.000 | 0.541 | 0.496 | 10.06 | 0.0% |

## Unanswerable-query breakdown

| System | Category | Queries | Abstention accuracy |
| --- | --- | ---: | ---: |
| Legacy-style TF-IDF proxy | adversarial_paraphrase | 6 | 0.0% |
| Legacy-style TF-IDF proxy | hard_negative | 8 | 0.0% |
| Legacy-style TF-IDF proxy | no_domain_anchor | 4 | 0.0% |
| Improved local SearchService | adversarial_paraphrase | 6 | 100.0% |
| Improved local SearchService | hard_negative | 8 | 100.0% |
| Improved local SearchService | no_domain_anchor | 4 | 100.0% |

## Method

- 40 corpus-supported questions and 4 audited source-gap questions use manually assigned, binary document-level qrels.
- 18 out-of-scope questions measure abstention accuracy.
- Query IDs are stored in the query CSV and validated across qrels and source-gap labels.
- Each system is initialized and receives one unmeasured warm-up query before timing.
- Up to 10 ranked chunks are requested; repeated filenames are collapsed at their first occurrence for document metrics.
- Failed queries remain in quality denominators with zero scores and are also reported as failures.
- Token-F1 and ROUGE-L are calculated only on corpus-supported questions and ignore numeric citation markers such as `[1]`.
- Q06, Q17, Q36, and Q44 are source gaps; their answer-overlap values remain in the detailed CSV but are excluded from aggregate answer quality.
- Latency percentiles include successful measured calls across supported, source-gap, and OOD queries after one unmeasured warm-up; initialization is excluded.
- This is an in-sample diagnostic set used during development, not a held-out estimate of generalization.

The detailed CSV should be inspected for category-specific wins, errors, abstentions, and confusable questions; aggregate scores alone do not establish factual correctness.
