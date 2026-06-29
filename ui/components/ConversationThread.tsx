"use client";

import { useEffect, useRef } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Message {
  id: string;
  role: "user" | "evie";
  text: string;
  /** true while the LLM is still streaming this message */
  streaming?: boolean;
}

interface ConversationThreadProps {
  messages: Message[];
  /** Show the STT "transcribing your voice" indicator */
  transcribing?: boolean;
}

// ── Typing dots (shown while Evie is thinking / STT is running) ───────────────

function TypingDots() {
  return (
    <span className="inline-flex items-end gap-[3px] h-4 ml-1" aria-label="…">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="block w-1 h-1 rounded-full bg-current animate-bounce"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}

// ── Single message bubble ─────────────────────────────────────────────────────

function Bubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <div className={["flex gap-3", isUser ? "flex-row-reverse" : "flex-row"].join(" ")}>
      {/* Avatar */}
      {!isUser && (
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-bold select-none">
          E
        </span>
      )}

      {/* Bubble */}
      <div
        className={[
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-sm"
            : "bg-muted text-foreground rounded-tl-sm",
        ].join(" ")}
      >
        {message.text}
        {message.streaming && <TypingDots />}
      </div>
    </div>
  );
}

// ── Thread ────────────────────────────────────────────────────────────────────

export function ConversationThread({
  messages,
  transcribing = false,
}: ConversationThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, transcribing]);

  if (messages.length === 0 && !transcribing) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center select-none py-16">
        <span className="text-3xl">👋</span>
        <p className="text-sm text-muted-foreground">
          Tap the mic and start talking.
          <br />
          Evie is listening.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {messages.map((msg) => (
        <Bubble key={msg.id} message={msg} />
      ))}

      {/* STT in-progress indicator */}
      {transcribing && (
        <div className="flex gap-3 flex-row-reverse">
          <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-primary/20 px-4 py-2.5 text-sm text-primary italic">
            Listening<TypingDots />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
