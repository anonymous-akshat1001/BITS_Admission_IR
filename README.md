# BITS Research Regulations Search

A local, explainable information-retrieval project for searching the 52 bundled BITS Pilani PhD and research-policy PDFs. It returns focused answers, page-level evidence, transparent ranking scores, and an explicit “not enough evidence” response when the corpus cannot support an answer.

The improved system runs without Qdrant, paid APIs, model downloads, or an internet connection after installation. A Gemini model can optionally rewrite retrieved evidence into a concise answer, but the deterministic extractive answerer is the default and fallback.

## Measured result

The diagnostic set contains 40 corpus-supported questions, four audited source-gap questions, and 18 deliberately unsupported questions (including eight hard negatives and six adversarial paraphrases that reuse domain vocabulary). The current deterministic evaluation produced:

| Metric | Legacy-style TF-IDF proxy | Improved system |
| --- | ---: | ---: |
| Document Hit@1 | 79.5% | **100.0%** |
| Document Hit@3 | 90.9% | **100.0%** |
| MRR@10 | 0.854 | **1.000** |
| nDCG@5 | 0.853 | **0.982** |
| Supported answer token-F1 | 0.215 | **0.381** |
| Supported answer ROUGE-L | 0.181 | **0.320** |
| Audited source-gap abstention | 0.0% | **100.0%** |
| Out-of-scope abstention | 0.0% | **100.0%** |
| Post-warm-up median latency | **0.21 ms** | 9.96 ms |
| Runtime failures | 0 | 0 |

These are in-sample diagnostic results on the small dataset used while tuning the rules, not a held-out or external estimate of generalization. The legacy number is a reproducible proxy because the original Qdrant index, credentials, and saved outputs were not committed. It preserves the old lossy preprocessing, 600-character chunks, and top-chunk answer behavior, but uses local unigram TF-IDF in place of the unavailable cloud/vector services.

Four reference answers (Q06, Q17, Q36, and Q44) contain policy details that are not fully present in the supplied PDFs. The improved system reports all four gaps instead of inventing answers and does not abstain on the 40 supported diagnostic questions. See [the evaluation report](artifacts/evaluation/summary.md) and [evaluation methodology](docs/EVALUATION.md).

## What the project does

The corpus covers research proposals, DRC/DAC procedures, qualifying examinations, fellowships, travel grants, sponsored-project rules, thesis submission, and related forms. It is not a general BITS admissions or campus-information search engine.

```text
52 PDFs
   ↓ page-aware extraction + reviewed OCR sidecars + table recovery
Unicode-safe cleaning, structure retention, domain normalization
   ↓
Page-bounded chunks with title, section, page, and stable source metadata
   ↓
BM25 + TF-IDF + phrase + proximity + title/scope signals
   ↓ diversity and overlap filtering
Focused extractive answer / safe abstention / optional grounded Gemini
   ↓
CLI, FastAPI, and Next.js evidence interface
```

### Retrieval pipeline

1. **Corpus loading:** PyMuPDF extracts all 52 PDFs and 171 pages. Repeated headers and footers, page numbers, broken hyphenation, and excess whitespace are cleaned without removing useful Unicode such as `₹`, bullets, or comparison symbols.
2. **Scanned documents:** two image-only policy PDFs use bundled, checksum-verified OCR sidecars that were visually checked against rendered pages. Every corpus document is searchable without requiring Tesseract at runtime.
3. **Chunking:** content is split within page boundaries at roughly 900 characters with 140 characters of overlap. Each chunk keeps its document title, section, page number, filename, and stable ID.
4. **Query processing:** phrases and notations such as “Ph.D.”, “full-time,” “T.A./D.A.,” “TA-DA,” Professional Development Fund, DRC, and DAC are normalized. Stopwords are removed, a conservative stemmer handles common variants, and a small transparent domain expansion map adds low-weight synonyms.
5. **Ranking:** normalized BM25 and TF-IDF scores are combined with phrase, proximity, title, and query-coverage signals. Scope checks distinguish easily confused policies such as national versus international travel, self-sponsored versus Institute fellowship, and proposal versus thesis documents.
6. **Result filtering:** near-duplicate chunks are suppressed and each document is limited to two results before fallback filling, so overlapping pages from one PDF do not dominate the list.
7. **Answering:** relevant sentences are selected and cited. Numeric/unit and explicit-identifier evidence improve answers for amounts, durations, percentages, counts, and acronyms; structured-table handlers preserve conditional PDF/DDF/CDF allocations and the interleaved CS/IS sub-area table. Full-corpus scope checks, direct-evidence checks, and audited source-gap rules refuse unsupported questions.

The method is deliberately understandable for a college IR project. It has no hidden training stage; the API returns the main additive score components, query matches, and final score for inspection.

## Quick start

Requirements:

- Python 3.11 recommended
- Node.js 18.17 or newer for the web interface

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
```

Inspect the corpus, run a query, and execute the tests:

```bash
make inspect
make search QUERY="What is the international travel award limit?"
make test
```

Start the API:

```bash
make api
```

The API is available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`. The first query builds the in-memory index and normally takes a few seconds; later queries are much faster.

In a second terminal, start the frontend:

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`. The interface shows answer confidence, warnings, processing time, matched terms, ranked evidence excerpts, and links to the cited PDF page.

Windows users can run the equivalent module commands after activating `.venv\Scripts\activate`; the Make targets use Unix-style virtual-environment paths.

## CLI and API usage

Direct CLI query:

```bash
.venv/bin/python -m backend.scripts.search \
  "How many credits are prescribed for a PhD student with a first degree?" \
  --top-k 5
```

API query:

```bash
curl -X POST http://localhost:8000/query/ \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the maximum National Travel Grant?","top_k":5,"answer_mode":"extractive"}'
```

`answer_mode` can be:

- `extractive`: deterministic local answer, requiring no key or network;
- `auto`: use Gemini when configured, otherwise use extractive mode;
- `gemini`: request Gemini and fall back safely if it is unavailable.

Every response includes the answer, confidence level, abstention state, citation ranks, retrieval method, processing time, warnings, and source records containing page numbers, excerpts, matched terms, component scores, and safe local PDF URLs. Older `/query/hybrid/` and `/query/hybrid-rerank/` clients remain compatible but now use the same improved service.

## Optional Gemini generation

The project does not need an LLM. If a Gemini API key and suitable quota are available, copy the example environment file and configure a model:

```bash
cp .env.example .env
# Edit GEMINI_API_KEY and, if needed, GEMINI_MODEL.
```

The default is `gemini-2.5-flash`. Any model name available to the configured Gemini account can be used. The prompt restricts generation to retrieved passages, requests a structured `answered`/`not_found` status and citations, preserves amounts and conditions, and treats source text as untrusted data. Invalid JSON, citation structures, status contradictions, or numeric claims absent from cited text fail closed instead of displaying an unverified generated answer; network/provider failures still use the local extractive fallback. Provider free-tier availability and quotas can change, so local extractive mode remains the fully free default.

## Evaluation

Run the complete comparison:

```bash
make evaluate
```

Inputs:

- `data/evaluation/queries.csv` — 44 stable IDs, questions, and reference answers;
- `data/evaluation/qrels.csv` — binary document-level relevance judgments;
- `data/evaluation/source_gaps.csv` — four manually audited missing-evidence labels;
- `data/evaluation/unanswerable.csv` — 18 unsupported questions across simple, hard-negative, and adversarial-paraphrase categories.

Outputs are regenerated in `artifacts/evaluation/`:

- `summary.md` — presentation-ready overall and category tables;
- `summary.json` — complete machine-readable results and corpus diagnostics;
- `per_query.csv` — query-by-query answers, rankings, metrics, latency, errors, and abstentions.

Retrieval is evaluated with Hit@1/3/5, MRR@10, and nDCG@5. Qrels for the four source gaps identify only the closest topical document, so those hits are not claims that the missing answer was retrieved. Token-F1 and ROUGE-L are averaged over the 40 supported questions; source-gap and out-of-scope abstention are reported separately. Failures remain in the denominator with a score of zero. Text overlap does not establish factual correctness, so the detailed CSV and cited pages should also be inspected.

## Project structure

```text
BITS_Admission_IR/
├── backend/
│   ├── api.py                  # FastAPI routes and response schema
│   ├── main_api.py             # Environment-aware API entry point
│   ├── ir_system/              # ingestion, preprocessing, ranking, answers
│   ├── scripts/                # search, corpus inspection, OCR, evaluation
│   └── tests/                  # unit, API, and corpus integration tests
├── data/
│   ├── evaluation/             # queries, qrels, unanswerable cases
│   └── ocr/                    # reviewed, checksum-bound OCR sidecars
├── document_corpus/            # 52 source PDFs
├── frontend/                   # Next.js evidence-search interface
├── artifacts/evaluation/       # reproducible benchmark outputs
├── docs/
│   ├── PROJECT_AUDIT.md        # documented original implementation
│   └── EVALUATION.md           # metric definitions and caveats
├── .env.example
└── Makefile
```

## Old behavior versus improved behavior

| Area | Original repository | Improved repository |
| --- | --- | --- |
| Runtime | Qdrant/cloud credentials and several model downloads required | Small in-memory CPU index; no service or model download |
| Preprocessing | Lowercased ASCII with page structure removed | Unicode-safe, page-aware, header/footer cleanup, headings and lists retained |
| Scans and tables | No working OCR or table recovery | Reviewed OCR sidecars and structured table extraction |
| Ranking | Opaque hybrid defaults; frontend skipped reranking | Explainable weighted signals, scope disambiguation, diversity filtering |
| Answer | First retrieved chunk returned verbatim | Focused cited sentences, numeric/table intent, explicit evidence gaps |
| Provenance | Local filesystem paths and no page citation | Safe PDF route, source title, page, section, excerpt, and score breakdown |
| Evaluation | Hard-coded path and only answer-text similarity | Portable qrels, standard IR metrics, per-category reports, failure and abstention metrics |
| Structure | Duplicate APIs/handlers and unsafe unused utilities | One service layer, thin API, focused scripts, pinned direct dependencies |
| UI | Generic branding, broken links, fixed non-rerank route | Responsive accessible search, health/retry states, warnings, ranked evidence |
| Tests | None | 62 backend unit/API/integration tests |

The exact pre-improvement audit is preserved in [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md).

## Remaining limitations and sensible next steps

- The relevance set has only 44 reference questions, binary document-level judgments, and was used during tuning. Add an independently written held-out set and page/passage-level graded judgments before making broader claims.
- Four reference answers are not fully supported by the current corpus. Add the missing official policy documents rather than encoding their answers in application logic.
- The PDFs are a static snapshot and may be outdated. Important eligibility, deadline, and financial information must be verified in the cited document and against the latest official notice.
- Retrieval is lexical and domain-tuned. It is transparent and fast, but a future experiment could compare it with one small local embedding model on the same qrels; keep the lexical system as a reproducible baseline.
- OCR sidecars are bound to exact PDF checksums. New image-only or changed PDFs require OCR generation followed by manual visual review.
- Optional Gemini generation introduces network, quota, latency, and model-availability dependencies. It should remain optional and must not replace source verification.
- The corpus does not include canonical public URLs for every PDF, so citations link to the bundled local copies.

These limitations are intentionally visible: the project favors demonstrable retrieval quality and trustworthy outputs over adding larger infrastructure or flashy features.
