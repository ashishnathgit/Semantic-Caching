import { useEffect, useState } from "react";
import { api } from "../api";
import type { ImportResult, ModelStatus } from "../types";

// Admin tools that change the model or the cache in bulk. Shown only to admins.
export function AdminTools({ onChanged }: { onChanged: () => Promise<void> | void }) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = () => api.modelStatus().then(setStatus).catch(() => undefined);
  useEffect(() => {
    load();
  }, []);
  // While a rebuild runs, poll until it is done, then refresh the page's data.
  useEffect(() => {
    if (status?.state !== "running") return;
    const timer = setInterval(async () => {
      const s = await api.modelStatus().catch(() => null);
      if (!s) return;
      setStatus(s);
      if (s.state !== "running") {
        setMsg(s.state === "done" ? `Model rebuilt in ${s.seconds?.toFixed(1)} s: k = ${s.k}, ${s.vocabulary_size} terms.` : null);
        if (s.state === "failed") setErr(`Rebuild failed: ${s.error}`);
        await onChanged();
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [status?.state, onChanged]);

  const [profileName, setProfileName] = useState("campus");
  const [csvText, setCsvText] = useState<string | null>(null);
  const [csvName, setCsvName] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [imported, setImported] = useState<ImportResult | null>(null);
  const [confirmForget, setConfirmForget] = useState(false);

  async function importCsv() {
    if (!csvText) return;
    setErr(null);
    setMsg(null);
    setImporting(true);
    try {
      setImported(await api.importCsv(profileName.trim(), csvText));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setImporting(false);
    }
  }

  async function forget() {
    setErr(null);
    try {
      const r = await api.forgetLlm();
      setMsg(`Removed ${r.removed} LLM-learned answers; ${r.entries} entries remain.`);
      setConfirmForget(false);
      await onChanged();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  async function rebuild() {
    setErr(null);
    setMsg(null);
    try {
      setStatus(await api.rebuildModel());
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  return (
    <section className="card space-y-5 p-5">
      <h2 className="m-0 text-[22px] font-semibold tracking-tight">Admin tools</h2>
      <div className="space-y-2">
        <h3 className="m-0 text-[15px] font-semibold">Rebuild model</h3>
        {status?.grows ? (
          <p className="m-0 max-w-prose text-[14px] leading-snug text-ink-soft">
            Refits TF-IDF + SVD on the FAQ texts plus every cached question with its answer, in the background, then
            re-embeds every stored question and swaps the model in one step. Now: k = {status.k},{" "}
            {status.vocabulary_size} terms, {status.documents} training texts; {status.new_since_build} new entries since the
            last fit{status.rebuild_every ? ` (automatic rebuild every ${status.rebuild_every})` : ""}.
          </p>
        ) : (
          <p className="m-0 max-w-prose text-[14px] leading-snug text-ink-soft">
            This profile's model is not fitted on cached answers (corpus.include_cached = false), so a rebuild would change
            nothing. Use the faq-live profile for a vocabulary that grows.
          </p>
        )}
        <button type="button" onClick={rebuild} disabled={!status?.grows || status.state === "running"}
          className="h-9 rounded-lg bg-ink px-4 text-[14px] font-medium text-sheet hover:bg-ink-soft disabled:opacity-40">
          {status?.state === "running" ? "Rebuilding…" : "Rebuild model"}
        </button>
      </div>
      <div className="space-y-2">
        <h3 className="m-0 text-[15px] font-semibold">Import FAQ CSV into a new profile</h3>
        <p className="m-0 max-w-prose text-[14px] leading-snug text-ink-soft">
          A CSV with the columns question and answer (optional: paraphrases, heldout) becomes a new profile with its own
          model and cache in artifacts/&lt;name&gt;/. The running profile and config.toml are not changed.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor="imp-name" className="text-[13px] text-ink-soft">Profile name</label>
          <input id="imp-name" value={profileName} onChange={(e) => setProfileName(e.target.value)}
            className="h-9 w-40 rounded-md border border-rule bg-sheet px-3 text-[14px] outline-none focus-visible:border-query" />
          <input type="file" accept=".csv,text/csv" aria-label="FAQ CSV file" className="text-[13px]"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              setImported(null);
              if (!f) return;
              setCsvName(f.name);
              setCsvText(await f.text());
            }} />
          <button type="button" onClick={importCsv} disabled={!csvText || !profileName.trim() || importing}
            className="h-9 rounded-lg bg-ink px-4 text-[14px] font-medium text-sheet hover:bg-ink-soft disabled:opacity-40">
            {importing ? "Importing…" : "Import"}
          </button>
        </div>
        {imported && (
          <p className="m-0 text-[14px] text-hit">
            Profile “{imported.profile}” built from {csvName}: {imported.entries} answers cached, k = {imported.k},{" "}
            {imported.vocabulary_size} terms. Serve it with <code className="num">{imported.serve}</code>.
          </p>
        )}
      </div>
      <div className="space-y-2">
        <h3 className="m-0 text-[15px] font-semibold">Forget LLM-learned answers</h3>
        <p className="m-0 max-w-prose text-[14px] leading-snug text-ink-soft">
          Removes every cached answer an LLM produced (including placeholders). FAQ answers and answers added by hand stay.
        </p>
        {confirmForget ? (
          <div className="flex gap-2">
            <button type="button" onClick={forget} className="h-9 rounded-lg bg-miss px-4 text-[14px] font-medium text-sheet">
              Yes, forget them
            </button>
            <button type="button" onClick={() => setConfirmForget(false)} className="h-9 rounded-lg border border-rule px-4 text-[14px]">
              Cancel
            </button>
          </div>
        ) : (
          <button type="button" onClick={() => setConfirmForget(true)}
            className="h-9 rounded-lg border border-rule px-4 text-[14px] text-ink hover:border-ink-faint">
            Forget LLM-learned rows
          </button>
        )}
      </div>
      {msg && <p className="m-0 text-[14px] text-hit">{msg}</p>}
      {err && <p role="alert" className="m-0 text-[14px] text-miss">{err}</p>}
    </section>
  );
}
