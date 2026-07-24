# BITS Research Regulations Search — Frontend

This is the Next.js interface for searching the BITS Pilani research-document
corpus. It is intentionally small: a single accessible search conversation,
grounded answers, and expandable evidence cards that link back to the cited PDF
page.

## Requirements

- Node.js 18.17 or newer
- The project backend running locally or at a reachable URL

## Setup

From this `frontend` directory:

```bash
npm install
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>. The example environment file points to the local
backend at `http://localhost:8000`; update it when the API is hosted elsewhere.
If the variable is omitted, the UI uses the same local URL as a development
fallback.

## API contract

The UI checks backend availability with:

```text
GET /health
```

Each search sends:

```http
POST /query/
Content-Type: application/json

{
  "query": "What is the international travel award limit?",
  "top_k": 5,
  "answer_mode": "auto"
}
```

The response must contain:

```json
{
  "answer": "The maximum award is ... [1]",
  "answer_type": "extractive",
  "confidence": "high",
  "abstained": false,
  "citations": [1],
  "retrieval_method": "BM25 + TF-IDF + phrase/title reranking",
  "processing_time_ms": 842.4,
  "source_documents": [
    {
      "rank": 1,
      "excerpt": "Relevant text from the source page...",
      "score": 0.88,
      "matched_terms": ["international", "travel", "award"],
      "metadata": {
        "doc_name": "travel-award.pdf",
        "title": "International Travel Award Guidelines",
        "page_start": 2,
        "page_end": 2,
        "source_url": "/documents/travel-award.pdf",
        "text_available": true
      }
    }
  ],
  "warnings": []
}
```

PDF links are opened through the backend using `source_url` and the source page
fragment. The backend should restrict that route to files in the approved corpus.

## Useful commands

```bash
npm run dev        # development server
npm run lint       # Next.js/ESLint checks
npm run typecheck  # TypeScript checks
npm run build      # production build
npm run start      # run a completed production build
```

## Interface behavior

- Enter submits a question; Shift+Enter adds a line.
- Example prompts fill the composer without sending automatically.
- A health indicator can be clicked to retry the backend check.
- Failed searches remain visible and can be retried.
- Answers show their mode, confidence, retrieval method, processing time,
  warnings, evidence excerpts, match terms, and page-specific PDF links.
- Conversation history is kept only in the current browser page and can be
  cleared from the header.

The interface reminds users to verify deadlines, eligibility, and financial
limits in the cited document because generated answers can still be incomplete.
