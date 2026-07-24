"use client";

import {
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  BookOpenCheck,
  CircleAlert,
  Eraser,
  RefreshCw,
  Send,
  UserRound,
} from "lucide-react";

import HomeStarter from "@/components/shared/HomeStarter";
import Loader from "@/components/shared/Loader";
import QueryResult from "@/components/shared/QueryResult";
import {
  ApiError,
  getHealth,
  searchRegulations,
  type QueryResponse,
} from "@/lib/api";

type ConnectionState = "checking" | "online" | "offline";

interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  text?: string;
  result?: QueryResponse;
  error?: string;
  retryQuery?: string;
}

function messageId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function connectionLabel(state: ConnectionState): string {
  if (state === "checking") return "Checking service";
  if (state === "online") return "Search service ready";
  return "Search service unavailable";
}

function shortConnectionLabel(state: ConnectionState): string {
  if (state === "checking") return "Checking";
  if (state === "online") return "Ready";
  return "Offline";
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const endOfConversationRef = useRef<HTMLDivElement>(null);
  const requestInFlightRef = useRef(false);

  const checkConnection = useCallback(async () => {
    setConnection("checking");
    try {
      const health = await getHealth();
      const status = health.status?.toLowerCase();
      setConnection(
        status === "error" || status === "unhealthy" || status === "degraded"
          ? "offline"
          : "online",
      );
    } catch {
      setConnection("offline");
    }
  }, []);

  useEffect(() => {
    void checkConnection();
  }, [checkConnection]);

  useEffect(() => {
    endOfConversationRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isLoading]);

  const submitQuery = async (rawQuery: string) => {
    const normalizedQuery = rawQuery.trim();
    if (!normalizedQuery || requestInFlightRef.current) return;

    requestInFlightRef.current = true;
    setMessages((current) => [
      ...current,
      { id: messageId(), role: "user", text: normalizedQuery },
    ]);
    setQuery("");
    setIsLoading(true);

    try {
      const result = await searchRegulations(normalizedQuery);
      setConnection("online");
      setMessages((current) => [
        ...current,
        { id: messageId(), role: "assistant", result },
      ]);
    } catch (error) {
      setConnection(
        error instanceof ApiError && !error.isConnectionError
          ? "online"
          : "offline",
      );
      const message =
        error instanceof ApiError
          ? error.message
          : "The search could not be completed. Please try again.";

      setMessages((current) => [
        ...current,
        {
          id: messageId(),
          role: "assistant",
          error: message,
          retryQuery: normalizedQuery,
        },
      ]);
    } finally {
      requestInFlightRef.current = false;
      setIsLoading(false);
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submitQuery(query);
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const selectPrompt = (prompt: string) => {
    setQuery(prompt);
    window.requestAnimationFrame(() => composerRef.current?.focus());
  };

  const clearConversation = () => {
    setMessages([]);
    setQuery("");
    window.requestAnimationFrame(() => composerRef.current?.focus());
  };

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-slate-50 text-slate-950">
      <a
        href="#main-content"
        className="sr-only z-50 rounded-md bg-white px-3 py-2 text-sm font-semibold text-slate-950 focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to search
      </a>

      <header className="z-20 shrink-0 border-b border-slate-200/90 bg-white/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-slate-950 text-orange-400">
              <BookOpenCheck aria-hidden="true" className="h-5 w-5" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-bold text-slate-950 sm:text-base">
                BITS Research Regulations Search
              </p>
              <p className="hidden text-xs text-slate-500 sm:block">
                Evidence-backed answers from the document corpus
              </p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => void checkConnection()}
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 transition hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 sm:px-3"
              aria-label={`${connectionLabel(connection)}. Check again.`}
            >
              <span
                aria-hidden="true"
                className={`h-2 w-2 rounded-full ${
                  connection === "online"
                    ? "bg-emerald-500"
                    : connection === "checking"
                      ? "animate-pulse bg-amber-400"
                      : "bg-rose-500"
                }`}
              />
              <span className="sm:hidden">{shortConnectionLabel(connection)}</span>
              <span className="hidden sm:inline">{connectionLabel(connection)}</span>
              <RefreshCw
                aria-hidden="true"
                className={`h-3.5 w-3.5 ${connection === "checking" ? "animate-spin" : ""}`}
              />
            </button>

            {messages.length > 0 && (
              <button
                type="button"
                onClick={clearConversation}
                disabled={isLoading}
                className="inline-flex h-8 w-8 items-center justify-center rounded-full text-slate-500 transition hover:bg-slate-100 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 disabled:cursor-not-allowed disabled:opacity-50"
                aria-label="Clear conversation"
                title="Clear conversation"
              >
                <Eraser aria-hidden="true" className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </header>

      <main id="main-content" className="min-h-0 w-full flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-6xl px-4 sm:px-6">
          {messages.length === 0 ? (
            <HomeStarter onSelectPrompt={selectPrompt} />
          ) : (
            <section
              className="mx-auto w-full max-w-4xl space-y-5 py-7 sm:py-10"
              aria-label="Search conversation"
              role="log"
              aria-live="polite"
              aria-relevant="additions"
              aria-busy={isLoading}
            >
              {messages.map((message) =>
                message.role === "user" ? (
                  <article key={message.id} className="flex justify-end" aria-label="Your question">
                    <div className="flex max-w-[92%] items-start gap-2.5 sm:max-w-[78%]">
                      <div className="rounded-2xl rounded-tr-md bg-slate-900 px-4 py-3 text-[15px] leading-6 text-white shadow-sm">
                        {message.text}
                      </div>
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-600 shadow-sm">
                        <UserRound aria-hidden="true" className="h-4 w-4" />
                      </span>
                    </div>
                  </article>
                ) : (
                  <article key={message.id} className="flex items-start gap-2.5" aria-label="Search answer">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-orange-100 text-orange-700">
                      {message.error ? (
                        <CircleAlert aria-hidden="true" className="h-4 w-4" />
                      ) : (
                        <BookOpenCheck aria-hidden="true" className="h-4 w-4" />
                      )}
                    </span>
                    <div className="min-w-0 flex-1 rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-4 shadow-sm sm:px-6 sm:py-5">
                      {message.result && <QueryResult result={message.result} />}
                      {message.error && (
                        <div role="alert">
                          <h2 className="text-sm font-semibold text-slate-900">Search unavailable</h2>
                          <p className="mt-1 text-sm leading-6 text-slate-600">{message.error}</p>
                          {message.retryQuery && (
                            <button
                              type="button"
                              onClick={() => void submitQuery(message.retryQuery || "")}
                              disabled={isLoading}
                              className="mt-3 inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                              <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
                              Try again
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                  </article>
                ),
              )}

              {isLoading && <Loader />}
              <div ref={endOfConversationRef} aria-hidden="true" />
            </section>
          )}
        </div>
      </main>

      <footer className="z-10 shrink-0 border-t border-slate-200 bg-white/95 backdrop-blur">
        <form
          onSubmit={handleSubmit}
          className="mx-auto w-full max-w-4xl px-4 py-3 sm:px-6 sm:py-4"
        >
          <label htmlFor="research-query" className="sr-only">
            Ask a question about BITS research regulations
          </label>
          <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition focus-within:border-orange-500 focus-within:ring-2 focus-within:ring-orange-100">
            <textarea
              ref={composerRef}
              id="research-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              rows={1}
              maxLength={500}
              disabled={isLoading}
              className="max-h-32 min-h-10 flex-1 resize-y bg-transparent px-2 py-2 text-[15px] leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed disabled:opacity-60"
              placeholder="Ask about fellowships, thesis submission, travel grants…"
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-orange-600 text-white shadow-sm transition hover:bg-orange-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300"
              aria-label="Send question"
            >
              <Send aria-hidden="true" className="h-4 w-4" />
            </button>
          </div>
          <div className="mt-1.5 flex items-center justify-between gap-3 px-1 text-[11px] text-slate-500">
            <span>Enter to search · Shift+Enter for a new line</span>
            <span>{query.length}/500</span>
          </div>
          <p className="mt-1 text-center text-[11px] leading-4 text-slate-500">
            Answers can be incomplete. Verify important information in the cited PDFs.
          </p>
        </form>
      </footer>
    </div>
  );
}
