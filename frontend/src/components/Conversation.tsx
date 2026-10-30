import { useEffect, useRef } from "react";
import type { MessageMeta, ThreadMessage } from "../types";
import { sourceLabel } from "./sources";

interface Props {
  messages: ThreadMessage[];
  pendingQuestion: string | null;
}

function howAnswered(meta: MessageMeta): { label: string; tone: string } {
  if (meta.hit) {
    return {
      label: meta.matched_query ? `From the cache · same as “${meta.matched_query}”` : "From the cache",
      tone: "text-hit",
    };
  }
  const by = meta.source === "ollama" ? `Ollama${meta.model ? ` (${meta.model})` : ""}` : sourceLabel(meta.source ?? "");
  if (meta.cache_used === false) {
    const n = meta.history_used ?? 0;
    return {
      label: `Follow-up · ${by} with ${n} earlier message${n === 1 ? "" : "s"} · not cached`,
      tone: "text-query",
    };
  }
  return { label: `New · ${by}${meta.stored_id ? " · now cached for everyone" : ""}`, tone: "text-miss" };
}

export function Conversation({ messages, pendingQuestion }: Props) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest" });
  }, [messages.length, pendingQuestion]);

  if (!messages.length && !pendingQuestion) {
    return (
      <div className="flex min-h-[10rem] items-center rounded-md border border-dashed border-rule px-5 py-6">
        <p className="m-0 max-w-prose text-[15px] leading-relaxed text-ink-soft">
          Ask something to start. Questions that stand on their own are checked against the shared cache.
          Follow-ups such as “what about the second one?” go to the model together with this conversation.
        </p>
      </div>
    );
  }

  return (
    <div className="max-h-[28rem] space-y-4 overflow-y-auto rounded-md border border-rule bg-sheet px-4 py-4" aria-live="polite">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <div key={i} className="flex justify-end">
            <p className="m-0 max-w-[85%] rounded-lg bg-paper px-3 py-2 text-[15px] leading-snug text-ink">{m.content}</p>
          </div>
        ) : (
          <div key={i} className="max-w-[92%]">
            <p className="m-0 whitespace-pre-wrap text-[15px] leading-[1.6] text-ink">{m.content}</p>
            <p className={`mt-1 mb-0 text-[12px] ${howAnswered(m.meta).tone}`}>{howAnswered(m.meta).label}</p>
          </div>
        ),
      )}
      {pendingQuestion && (
        <>
          <div className="flex justify-end">
            <p className="m-0 max-w-[85%] rounded-lg bg-paper px-3 py-2 text-[15px] leading-snug text-ink">
              {pendingQuestion}
            </p>
          </div>
          <p className="m-0 text-[14px] text-ink-soft">Thinking…</p>
        </>
      )}
      <div ref={end} />
    </div>
  );
}
