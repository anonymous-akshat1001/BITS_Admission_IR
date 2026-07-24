# Project Audit: Original System

> **Audit scope:** This document records the repository as it existed **before the July 2026 improvement**. It is a baseline description, not documentation of the improved implementation.

## 1. Project goal

The original project was intended to help BITS Pilani students, research scholars, and faculty find answers in institutional PhD and research documents. Its corpus covers research regulations, DRC procedures, fellowships, travel grants, thesis forms, proposal formats, and GCIR processes. Despite the repository and frontend using the phrase "Admission IR" in places, the implemented scope was primarily **BITS Pilani research regulations and policies**, not general admissions information.

The proposed design combined dense semantic retrieval, sparse term retrieval, cross-encoder reranking, and retrieval-augmented answer generation with source attribution. In the audited code, retrieval and reranking were implemented, but LLM answer generation was disabled.

## 2. Verified baseline inventory

| Item | Original state |
| --- | ---: |
| PDF corpus | 52 files |
| Total PDF pages | 171 pages |
| Corpus size on disk | Approximately 14 MB |
| Evaluation data | 44 question-answer pairs |
| Vector store | Qdrant collection `hybrid_corpus_v1` |

The PDFs were stored in `document_corpus/`. The evaluation file, `IR_test_dataset.csv`, contained two columns: `Question` and `Answer`. It did not contain expected source documents, relevant passages, page numbers, relevance grades, or unanswerable examples.

## 3. Original end-to-end workflow

1. `backend/document_processing_hybrid.py` enumerated PDF files and compared their modification time and size with a JSON checkpoint.
2. `PyMuPDFLoader` extracted text from each page.
3. All pages from one PDF were joined into one text string and cleaned.
4. A recursive character splitter produced overlapping chunks.
5. Each chunk received one dense embedding and one sparse embedding.
6. Both named vectors, the cleaned chunk text, and document-level metadata were stored in Qdrant.
7. At query time, LangChain's Qdrant hybrid retrieval combined dense and sparse search.
8. The optional reranking endpoint passed a larger candidate set through a cross-encoder.
9. The first returned chunk was used verbatim as the API's `answer`.
10. The Next.js frontend displayed the answer and expandable source chunks.

## 4. Input and preprocessing

### Document input

The ingestion pipeline accepted PDF files from `document_corpus/`. Metadata attached to every chunk consisted mainly of:

- local file path;
- document filename;
- loader `source` value, normally another local path;
- filesystem modification date;
- total number of pages in the PDF.

Page-level provenance, official source URLs, section titles, chunk numbers, effective dates, and policy versions were not retained.

### Text preprocessing

The original `clean_text()` function:

- converted all text to lowercase;
- removed ASCII control characters;
- removed every non-ASCII character;
- collapsed whitespace;
- replaced underscores with spaces.

It did not implement the Unicode normalization, header/footer removal, layout preservation, OCR fallback, table handling, or figure handling described in the README. Because newline control characters were removed before chunking, most configured heading and newline separators were ineffective. The indexed text was also reused for display, so retrieved answers appeared lowercased and lost document formatting.

### Chunking

Documents were split with `RecursiveCharacterTextSplitter` using:

- chunk size: 600 characters;
- overlap: 150 characters;
- separators for headings, blank lines, sentences, and spaces.

Chunk size was character-based rather than token-based. All pages had already been merged, so a chunk could cross a page boundary and could not cite its actual page. There was no minimum-content check, boilerplate removal, near-duplicate filtering, or stable chunk identifier.

## 5. Indexing, retrieval, and ranking

### Indexing

The original index used:

- dense model: `BAAI/bge-small-en-v1.5`;
- dense dimension: 384;
- dense distance: cosine similarity;
- sparse model: `prithivida/Splade_PP_en_v1` (SPLADE++);
- vector database: Qdrant with named `dense-vector` and `sparse-vector` fields.

Dense embeddings were generated as a group. Sparse embeddings were generated one chunk at a time; a failed sparse embedding was replaced with an empty sparse vector. Points used random UUIDs. The code checked whether the collection existed but did not verify that an existing collection had the expected vector schema.

### Hybrid retrieval

`/query/hybrid/` requested the top 10 results through `QdrantVectorStore` in `RetrievalMode.HYBRID`. Fusion behavior was left to the installed LangChain/Qdrant default; no explicit fusion weights or reproducible fusion configuration appeared in code. Queries received no normalization, domain acronym expansion, filtering, score threshold, or out-of-domain detection.

### Reranking

`/query/hybrid-rerank/` retrieved 20 hybrid candidates and reranked them with `cross-encoder/ms-marco-MiniLM-L-6-v2`, returning the top 3. There was no overlap deduplication or document diversity step, so adjacent chunks from one source could occupy several result positions. Retrieval and reranker scores were not dependably included in API responses.

## 6. Original evaluation approach

`backend/evaluate_api.py` sent every test question to either the hybrid or hybrid-rerank API and calculated:

- sentence BLEU;
- ROUGE-L F1;
- BERTScore F1;
- request latency;
- peak memory observed in the evaluation client process.

Results were intended to be written to a mode-specific CSV. This measured textual overlap between each reference answer and the returned raw chunk, rather than retrieval relevance. It did not measure Precision@k, Recall@k, Hit Rate, MRR, MAP, or nDCG, because the dataset had no relevance judgments. Client-process memory was not a measurement of backend memory, and cold-start model loading was not separated from warm-query latency.

## 7. Output, API, and UI structure

The active FastAPI application was `backend/main_api.py`. It exposed:

- `GET /` for a status summary;
- `GET /health` for lightweight component flags;
- `POST /query/hybrid/`;
- `POST /query/hybrid-rerank/`.

Both query endpoints accepted JSON shaped as `{"query": "..."}` and returned:

```json
{
  "answer": "top retrieved chunk",
  "source_documents": [
    {
      "page_content": "retrieved chunk",
      "metadata": {
        "doc_name": "source.pdf",
        "file_path": "local/path/source.pdf",
        "source": "local/path/source.pdf"
      }
    }
  ]
}
```

The `answer` was not generated or synthesized: it was the first chunk returned by the selected retriever. The fallback "I don't know" response occurred only when retrieval returned no documents, which is uncommon for a non-empty nearest-neighbor index.

The frontend was a Next.js 14 chat page in `frontend/app/page.tsx`. It read `NEXT_PUBLIC_API_URL`, displayed user and bot messages, rendered answers as Markdown, and placed retrieved chunks in expandable accordions. Source links used the backend's `source` metadata, which was normally a server-local path and therefore not useful to a browser. Although code selected between two endpoints, `useReranker` was fixed to `false`, so the visible UI always used the non-reranked hybrid endpoint. Frontend titles referred broadly to "BITS Admission" or "BITS Pilani information," which did not accurately communicate the research-policy corpus.

## 8. Major weaknesses in the original system

1. **No active answer-generation stage.** LLM chains were commented out, so the product labeled as RAG returned a raw chunk.
2. **Weak presentation and abstention.** Answers were lowercased fragments, and unrelated queries still received a nearest result because no confidence threshold existed.
3. **Lost page and document structure.** Pages were merged, formatting was removed, and usable page citations were impossible.
4. **Risky checkpoint semantics.** Checkpoints were saved before splitting, embedding, or indexing succeeded. A failed run could subsequently skip unindexed files.
5. **Stale-document bug.** Deleted source PDFs were not reliably detected or removed from Qdrant, leaving obsolete chunks searchable.
6. **Insufficient metadata.** There were no official URLs, page ranges, section names, stable IDs, document versions, or effective dates.
7. **Untuned, opaque ranking.** Fusion defaults, candidate counts, chunk sizes, and reranker depth had not been justified through retrieval metrics.
8. **No duplicate control.** Overlapping chunks and similar policy/form versions could dominate rankings or provide conflicting information.
9. **Evaluation did not assess IR quality.** It measured answer-text similarity without qrels and omitted standard ranked-retrieval metrics.
10. **Evaluation was not portable.** Its dataset path was hard-coded to a Windows user directory, and the documented command omitted the required mode argument.
11. **Code duplication.** Multiple API and query-handler files implemented overlapping, inconsistent versions of the same pipeline; `openai_main_api.py` did not actually use OpenAI.
12. **Configuration and dependency problems.** Requirements were unpinned, some imported packages were undeclared, and the active API forced a Render-specific model-cache path.
13. **Unreliable health reporting and initialization.** Creating a Qdrant client object was treated as connectivity, and concurrent first requests could trigger duplicate lazy model loading.
14. **Unsafe corpus utility.** The DOCX conversion path wrote an OOXML document with a `.pdf` extension and could then delete the source file.
15. **No backend tests.** Cleaning, chunking, checkpoint recovery, fusion, reranking, API validation, and evaluation behavior had no automated coverage.

## 9. Baseline reproducibility limitation

No trustworthy numerical baseline could be reproduced from the repository alone before the July 2026 improvement. The repository did not include Qdrant credentials, a local index snapshot, generated evaluation result files, pinned Python dependencies, or source-level relevance judgments. In addition, the evaluation script's absolute Windows dataset path prevented it from running as checked in.

Therefore, the pre-improvement baseline is documented here primarily through verified corpus inventory and static code inspection. Any later before/after score table should clearly state whether the "before" figures were newly rerun from the preserved original implementation or whether they came from a different environment; figures should not be presented as historical measurements unless they are reproducible.
