import { BookOpenCheck, FileSearch, ShieldCheck } from "lucide-react";

const EXAMPLE_QUERIES = [
  "What is the maximum grant for the International Travel Award?",
  "Who is eligible for an institute fellowship?",
  "What documents are required for PhD thesis submission?",
  "How is a Doctoral Advisory Committee constituted?",
];

interface HomeStarterProps {
  onSelectPrompt: (prompt: string) => void;
}

export default function HomeStarter({ onSelectPrompt }: HomeStarterProps) {
  return (
    <section className="mx-auto w-full max-w-4xl py-8 sm:py-14" aria-labelledby="welcome-title">
      <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 px-6 py-9 text-white sm:px-10 sm:py-12">
          <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl bg-orange-500 shadow-lg shadow-orange-950/20">
            <BookOpenCheck aria-hidden="true" className="h-6 w-6" />
          </div>
          <p className="mb-2 text-sm font-semibold uppercase tracking-[0.16em] text-orange-300">
            BITS Pilani research support
          </p>
          <h1 id="welcome-title" className="max-w-2xl text-3xl font-bold tracking-tight sm:text-4xl">
            Find answers in research regulations and policies
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
            Search PhD procedures, fellowships, travel grants, thesis requirements,
            and related research documents. Every answer includes the retrieved
            evidence for verification.
          </p>
        </div>

        <div className="grid gap-6 px-6 py-7 sm:px-10 sm:py-9 lg:grid-cols-[1fr_220px]">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Try an example</h2>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {EXAMPLE_QUERIES.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => onSelectPrompt(prompt)}
                  className="group flex min-h-20 items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-left text-sm font-medium leading-5 text-slate-700 transition hover:border-orange-300 hover:bg-orange-50 hover:text-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2"
                >
                  <FileSearch
                    aria-hidden="true"
                    className="mt-0.5 h-4 w-4 shrink-0 text-orange-600"
                  />
                  <span>{prompt}</span>
                </button>
              ))}
            </div>
          </div>

          <aside className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-4 text-sm text-emerald-950">
            <div className="flex items-center gap-2 font-semibold">
              <ShieldCheck aria-hidden="true" className="h-4 w-4" />
              Verify important details
            </div>
            <p className="mt-2 leading-6 text-emerald-900/80">
              Open the cited PDF before acting on deadlines, eligibility rules, or
              financial limits.
            </p>
          </aside>
        </div>
      </div>
    </section>
  );
}
