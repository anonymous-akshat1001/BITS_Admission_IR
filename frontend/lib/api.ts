const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();

export const API_BASE_URL = (configuredApiUrl || "http://localhost:8000").replace(
  /\/+$/,
  "",
);

const HEALTH_TIMEOUT_MS = 8_000;
const QUERY_TIMEOUT_MS = 75_000;

export interface SourceDocumentMetadata {
  doc_name: string;
  title: string;
  page_start: number | null;
  page_end: number | null;
  source_url: string | null;
  text_available: boolean;
}

export interface SourceDocument {
  rank: number;
  excerpt: string;
  score: number | null;
  matched_terms: string[];
  metadata: SourceDocumentMetadata;
}

export interface QueryResponse {
  answer: string;
  answer_type: string;
  confidence: string;
  abstained: boolean;
  citations: number[];
  retrieval_method: string;
  processing_time_ms: number;
  source_documents: SourceDocument[];
  warnings: string[];
}

export interface HealthResponse {
  status?: string;
  [key: string]: unknown;
}

export class ApiError extends Error {
  readonly isConnectionError: boolean;

  constructor(message: string, isConnectionError = false) {
    super(message);
    this.name = "ApiError";
    this.isConnectionError = isConnectionError;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isSourceDocument(value: unknown): value is SourceDocument {
  if (!isRecord(value) || !isRecord(value.metadata)) return false;

  const metadata = value.metadata;
  return (
    typeof value.rank === "number" &&
    typeof value.excerpt === "string" &&
    isNullableNumber(value.score) &&
    Array.isArray(value.matched_terms) &&
    value.matched_terms.every((term) => typeof term === "string") &&
    typeof metadata.doc_name === "string" &&
    typeof metadata.title === "string" &&
    isNullableNumber(metadata.page_start) &&
    isNullableNumber(metadata.page_end) &&
    isNullableString(metadata.source_url) &&
    typeof metadata.text_available === "boolean"
  );
}

function isQueryResponse(value: unknown): value is QueryResponse {
  return (
    isRecord(value) &&
    typeof value.answer === "string" &&
    typeof value.answer_type === "string" &&
    typeof value.confidence === "string" &&
    typeof value.abstained === "boolean" &&
    Array.isArray(value.citations) &&
    value.citations.every((citation) => typeof citation === "number") &&
    typeof value.retrieval_method === "string" &&
    typeof value.processing_time_ms === "number" &&
    Number.isFinite(value.processing_time_ms) &&
    Array.isArray(value.source_documents) &&
    value.source_documents.every(isSourceDocument) &&
    Array.isArray(value.warnings) &&
    value.warnings.every((warning) => typeof warning === "string")
  );
}

function errorMessage(payload: unknown, fallback: string): string {
  if (!isRecord(payload)) return fallback;

  if (typeof payload.detail === "string") return payload.detail;
  if (typeof payload.message === "string") return payload.message;

  if (Array.isArray(payload.detail)) {
    const messages = payload.detail
      .map((item) => {
        if (!isRecord(item) || typeof item.msg !== "string") return null;
        return item.msg;
      })
      .filter((item): item is string => Boolean(item));

    if (messages.length > 0) return messages.join("; ");
  }

  return fallback;
}

async function fetchJson(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<unknown> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
    const payload: unknown = await response.json().catch(() => null);

    if (!response.ok) {
      throw new ApiError(
        errorMessage(payload, `The server returned ${response.status}.`),
      );
    }

    return payload;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        "The request took too long. The models may still be starting; please try again.",
        true,
      );
    }
    if (error instanceof ApiError) throw error;

    throw new ApiError(
      "The search service could not be reached. Check that the backend is running.",
      true,
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const payload = await fetchJson(
    "/health",
    { method: "GET", cache: "no-store" },
    HEALTH_TIMEOUT_MS,
  );

  return isRecord(payload) ? payload : {};
}

export async function searchRegulations(query: string): Promise<QueryResponse> {
  const payload = await fetchJson(
    "/query/",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: 5,
        answer_mode: "auto",
      }),
    },
    QUERY_TIMEOUT_MS,
  );

  if (!isQueryResponse(payload)) {
    throw new ApiError("The server returned an unexpected response format.");
  }

  return payload;
}

export function buildPdfUrl(
  sourceUrl: string | null,
  pageStart: number | null,
): string | null {
  if (!sourceUrl) return null;

  const normalizedPath = sourceUrl.startsWith("/")
    ? sourceUrl
    : `/${sourceUrl}`;
  const pageFragment = pageStart && pageStart > 0 ? `#page=${pageStart}` : "";

  return `${API_BASE_URL}${normalizedPath}${pageFragment}`;
}
