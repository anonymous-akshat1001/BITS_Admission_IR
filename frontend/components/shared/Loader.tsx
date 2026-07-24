import { LoaderCircle, Search } from "lucide-react";

export default function Loader() {
  return (
    <div className="flex items-center gap-3" role="status" aria-live="polite">
      <span className="flex h-9 w-9 items-center justify-center rounded-full bg-orange-100 text-orange-700">
        <Search aria-hidden="true" className="h-4 w-4" />
      </span>
      <div className="flex items-center gap-2 rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-600 shadow-sm">
        <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin text-orange-600" />
        Searching the regulations…
      </div>
    </div>
  );
}
