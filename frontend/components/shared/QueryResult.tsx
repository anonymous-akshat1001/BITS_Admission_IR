import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertTriangle,
  ChevronDown,
  ExternalLink,
  FileText,
} from "lucide-react";

import {
  buildPdfUrl,
  type QueryResponse,
  type SourceDocument,
} from "@/lib/api";

interface QueryResultProps {
  result: QueryResponse;
}

function humanize(value: string): string {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function documentTitle(source: SourceDocument): string {
  if (source.metadata.title?.trim()) return source.metadata.title;

  return (
    humanize(source.metadata.doc_name.replace(/\.pdf$/i, "")) ||
    "Source document"
  );
}

function pageLabel(source: SourceDocument): string | null {
  const { page_start: start, page_end: end } = source.metadata;
  if (!start) return null;
  if (!end || start === end) return `Page ${start}`;
  return `Pages ${start}–${end}`;
}

function scoreLabel(score: number | null): string | null {
  if (score === null || !Number.isFinite(score)) return null;
  if (score >= 0 && score <= 1) return `${Math.round(score * 100)}% match`;
  return `Score ${score.toFixed(3)}`;
}

function answerTypeLabel(answerType: string, abstained: boolean): string {
  if (abstained) return "Insufficient evidence";
  const normalized = answerType.toLowerCase();
  if (normalized === "generated") return "Grounded answer";
  if (normalized === "extractive") return "Retrieved extract";
  if (normalized === "no_answer") return "No supported answer";
  return humanize(answerType);
}

function answerTypeClass(answerType: string, abstained: boolean): string {
  if (abstained) return "bg-amber-50 text-amber-800";
  const normalized = answerType.toLowerCase();
  if (normalized === "no_answer") return "bg-amber-50 text-amber-800";
  if (normalized === "extractive") return "bg-sky-50 text-sky-800";
  return "bg-emerald-50 text-emerald-800";
}

function SourceCard({ source, index }: { source: SourceDocument; index: number }) {
  const title = documentTitle(source);
  const pages = pageLabel(source);
  const score = scoreLabel(source.score);
  const pdfUrl = buildPdfUrl(source.metadata.source_url, source.metadata.page_start);

  return (
    <details
      id={`source-${source.rank || index + 1}`}
      className="group scroll-mt-24 rounded-xl border border-slate-200 bg-slate-50/70 open:bg-white"
      open={index === 0}
    >
      <summary className="flex cursor-pointer list-none items-start gap-3 rounded-xl px-4 py-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-orange-100 text-xs font-bold text-orange-800">
          {source.rank || index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span
            className="block truncate text-sm font-semibold text-slate-900"
            title={title}
          >
            {title}
          </span>
          <span className="mt-0.5 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500">
            {pages && <span>{pages}</span>}
            {score && <span>{score}</span>}
            {!source.metadata.text_available && <span>Limited text</span>}
          </span>
        </span>
        <ChevronDown
          aria-hidden="true"
          className="mt-1 h-4 w-4 shrink-0 text-slate-500 transition-transform group-open:rotate-180"
        />
      </summary>

      <div className="border-t border-slate-200 px-4 py-4">
        <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">
          {source.excerpt || "No text excerpt is available for this source."}
        </p>

        {source.matched_terms.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5" aria-label="Matched terms">
            {source.matched_terms.slice(0, 8).map((term, termIndex) => (
              <span
                key={`${term}-${termIndex}`}
                className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[11px] font-medium text-slate-600"
              >
                {term}
              </span>
            ))}
          </div>
        )}

        <div className="mt-4">
          {pdfUrl ? (
            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-sm font-semibold text-orange-700 underline-offset-4 hover:underline"
            >
              Open PDF{pages ? ` · ${pages}` : ""}
              <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
              <span className="sr-only"> for {title}</span>
            </a>
          ) : (
            <span className="text-xs text-slate-500">PDF link unavailable</span>
          )}
        </div>
      </div>
    </details>
  );
}

export default function QueryResult({ result }: QueryResultProps) {
  const confidence = result.confidence.trim()
    ? `${humanize(result.confidence)} confidence`
    : null;
  const processingTime = Number.isFinite(result.processing_time_ms)
    ? `${(result.processing_time_ms / 1000).toFixed(1)}s`
    : null;
  const availableRanks = new Set(
    result.source_documents.map((source, index) => source.rank || index + 1),
  );
  const answerWithCitationLinks = result.answer.replace(
    /\[(\d+)\](?!\()/g,
    (citation, rankText: string) => {
      const rank = Number(rankText);
      return availableRanks.has(rank)
        ? `[[${rank}]](#source-${rank})`
        : citation;
    },
  );

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs font-medium text-slate-500">
        <span
          className={`rounded-full px-2.5 py-1 ${answerTypeClass(result.answer_type, result.abstained)}`}
        >
          {answerTypeLabel(result.answer_type, result.abstained)}
        </span>
        {confidence && <span>{confidence}</span>}
        {result.retrieval_method && (
          <span>via {humanize(result.retrieval_method)}</span>
        )}
        {processingTime && <span>{processingTime}</span>}
      </div>

      <div className="answer-content text-[15px] leading-7 text-slate-800">
        <Markdown remarkPlugins={[remarkGfm]}>{answerWithCitationLinks}</Markdown>
      </div>

      {result.warnings.length > 0 && (
        <div
          className="mt-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5 text-sm text-amber-900"
          role="note"
        >
          <div className="flex gap-2">
            <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="space-y-1">
              {result.warnings.map((warning, index) => (
                <p key={`${warning}-${index}`}>{warning}</p>
              ))}
            </div>
          </div>
        </div>
      )}

      <section className="mt-5 border-t border-slate-200 pt-4" aria-label="Sources">
        <div className="mb-3 flex items-center gap-2">
          <FileText aria-hidden="true" className="h-4 w-4 text-slate-500" />
          <h3 className="text-sm font-semibold text-slate-900">
            Sources ({result.source_documents.length})
          </h3>
        </div>

        {result.source_documents.length > 0 ? (
          <div className="space-y-2">
            {result.source_documents.map((source, index) => (
              <SourceCard
                key={`${source.metadata.doc_name}-${source.metadata.page_start}-${source.rank}-${index}`}
                source={source}
                index={index}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">
            No supporting source passed the relevance threshold.
          </p>
        )}
      </section>
    </div>
  );
}
