import { useState } from "react";

interface Props {
  onSignIn: (name: string, adminKey: string | null) => Promise<void>;
  adminEnabled: boolean; // the server has SEMCACHE_ADMIN_KEY set
}

const NAME_RE = /^[a-z0-9][a-z0-9_.-]{0,31}$/;

export function SignIn({ onSignIn, adminEnabled }: Props) {
  const [name, setName] = useState("");
  const [adminKey, setAdminKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const clean = name.trim().toLowerCase();
    if (!NAME_RE.test(clean)) {
      setError("Use 1-32 letters, digits, dots, dashes or underscores, starting with a letter or digit.");
      return;
    }
    setPending(true);
    try {
      await onSignIn(clean, showKey && adminKey.trim() ? adminKey.trim() : null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="mx-auto max-w-[30rem] rounded-lg border border-rule bg-sheet px-6 py-8 sm:px-8">
      <h2 className="mt-0 mb-2 font-math text-[24px] font-normal">Who's asking?</h2>
      <p className="mt-0 mb-6 text-[15px] leading-relaxed text-ink-soft">
        Pick a name to keep your conversations and the answers you add apart from everyone else's. The
        cache itself is shared: a question someone else already asked is answered instantly for you too.
      </p>
      <form onSubmit={submit} className="space-y-3" noValidate>
        <label htmlFor="user-name" className="block text-[14px] text-ink-soft">
          Your name
        </label>
        <div className="flex gap-2">
          <input
            id="user-name"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError(null);
            }}
            placeholder="e.g. priya"
            autoComplete="username"
            autoFocus
            className="h-11 min-w-0 flex-1 rounded-md border border-ink-faint bg-sheet px-4 text-[16px] text-ink outline-none placeholder:text-ink-faint focus-visible:border-query focus-visible:ring-2 focus-visible:ring-query/30"
          />
          <button
            type="submit"
            disabled={pending}
            className="h-11 rounded-md bg-query px-5 text-[15px] font-medium text-sheet hover:bg-ink disabled:opacity-60"
          >
            Continue
          </button>
        </div>
        {error && (
          <p role="alert" className="m-0 text-[13px] text-miss">
            {error}
          </p>
        )}
        <p className="m-0 text-[12px] leading-snug text-ink-faint">
          There's no password: anyone who types the same name sees the same conversations.
        </p>
        {showKey ? (
          <div className="space-y-1 pt-2">
            <label htmlFor="admin-key" className="block text-[14px] text-ink-soft">
              Admin key
            </label>
            <input
              id="admin-key"
              type="password"
              value={adminKey}
              onChange={(e) => {
                setAdminKey(e.target.value);
                setError(null);
              }}
              autoComplete="current-password"
              className="h-11 w-full rounded-md border border-ink-faint bg-sheet px-4 text-[16px] text-ink outline-none focus-visible:border-query focus-visible:ring-2 focus-visible:ring-query/30"
            />
            <p className="m-0 text-[12px] leading-snug text-ink-faint">
              Kept only in this browser tab. Lets you reset the shared cache and add or delete shared answers.
            </p>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setShowKey(true)}
            className="p-0 text-[13px] text-query underline-offset-2 hover:underline"
          >
            {adminEnabled ? "I have an admin key" : "Admin key (admin actions are turned off on this server)"}
          </button>
        )}
      </form>
    </section>
  );
}
