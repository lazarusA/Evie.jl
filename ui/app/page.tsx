"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ModeToggle, type Mode } from "@/components/ModeToggle";
import { SettingsBar, type AssistantSettings } from "@/components/SettingsBar";
import { MicButton, type MicState } from "@/components/MicButton";
import { VoiceBar } from "@/components/VoiceBar";
import { ConversationThread, type Message } from "@/components/ConversationThread";
import { AudioUploadButton } from "@/components/AudioUploadButton";
import { ThemeToggle } from "@/components/ThemeToggle"

// ── Demo conversation pairs ───────────────────────────────────────────────────

const DEMO_PAIRS: { user: string; evie: string }[] = [
  {
    user: "What's the capital of France?",
    evie: "Paris is the capital of France. It's been the country's political and cultural centre for over a thousand years.",
  },
  {
    user: "Can you write me a short poem about the sea?",
    evie: "Sure —\n\nThe sea forgets nothing,\nit swallows the light\nand gives it back changed,\ncolder, more bright.\n\nThe shore keeps its counsel,\nthe tide keeps its word —\nand everything spoken\nto the water is heard.",
  },
  {
    user: "Summarise the theory of relativity in one sentence.",
    evie: "Einstein's theory of relativity tells us that space and time are woven into a single fabric that's warped by mass and energy, which is why gravity bends light and clocks tick slower near heavy objects.",
  },
];

let demoIdx = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

function uid() {
  return Math.random().toString(36).slice(2, 9);
}

// Simulate streaming text character by character
function streamText(
  text: string,
  onChunk: (partial: string) => void,
  onDone: () => void,
  delay = 18
) {
  let i = 0;
  const tick = setInterval(() => {
    i++;
    onChunk(text.slice(0, i));
    if (i >= text.length) {
      clearInterval(tick);
      onDone();
    }
  }, delay);
  return () => clearInterval(tick);
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Home() {
  const [micState, setMicState]     = useState<MicState>("idle");
  const [transcribing, setTranscribing] = useState(false);
  const [messages, setMessages]     = useState<Message[]>([]);
  const [mode, setMode]             = useState<Mode>("balanced");
  const [settings, setSettings]     = useState<AssistantSettings>({
    stt:    "whisper-small",
    llm:    "llama-3.1-8b",
    device: "cpu",
  });

  const cleanupRef = useRef<(() => void) | null>(null);
  const micTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Simulate the full turn: listen → STT → LLM stream
  const runTurn = useCallback(() => {
    const pair = DEMO_PAIRS[demoIdx % DEMO_PAIRS.length];
    demoIdx++;

    // 1. Fake STT: show "listening" for 1.5s then resolve user text
    setTranscribing(true);
    micTimerRef.current = setTimeout(() => {
      setTranscribing(false);
      setMicState("idle");

      // Add user message
      const userMsg: Message = { id: uid(), role: "user", text: pair.user };
      setMessages((prev) => [...prev, userMsg]);

      // 2. Fake LLM stream
      const evieId = uid();
      setMessages((prev) => [
        ...prev,
        { id: evieId, role: "evie", text: "", streaming: true },
      ]);

      cleanupRef.current = streamText(
        pair.evie,
        (partial) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === evieId ? { ...m, text: partial } : m))
          );
        },
        () => {
          setMessages((prev) =>
            prev.map((m) => (m.id === evieId ? { ...m, streaming: false } : m))
          );
        },
        mode === "fast" ? 10 : mode === "thorough" ? 28 : 18
      );
    }, 1500);
  }, [mode]);

  const handleMicToggle = useCallback(() => {
    if (micState === "listening") {
      // Cancel mid-listen
      if (micTimerRef.current) clearTimeout(micTimerRef.current);
      setMicState("idle");
      setTranscribing(false);
    } else {
      setMicState("listening");
      runTurn();
    }
  }, [micState, runTurn]);

  // File upload: treat as a completed "recording" and run the same STT→LLM turn
  const handleFileUpload = useCallback((file: File) => {
    if (micState === "listening") return; // ignore if already recording
    setMicState("listening");             // reuse listening state for the status dot
    runTurn();
    // In production: pass `file` to the Julia STT backend here
    void file;
  }, [micState, runTurn]);

  // Cleanup on unmount
  useEffect(() => () => {
    cleanupRef.current?.();
    if (micTimerRef.current) clearTimeout(micTimerRef.current);
  }, []);

  return (
    <div className="min-h-svh bg-background text-foreground flex flex-col items-center">
      <div className="w-full max-w-xl flex flex-col min-h-svh px-4 py-6 gap-4 sm:py-8">

        {/* ── Header ───────────────────────────────────────────────────────── */}
        <header className="flex items-center justify-between">
          <div className="flex flex-col leading-none">
            <span className="text-base font-bold tracking-tight">Evie</span>
            <span className="text-[11px] text-muted-foreground font-medium tracking-wide">
              offline · Evie.jl
            </span>
          </div>
          {/* Status dot + theme toggle */}
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span
                className={[
                  "h-1.5 w-1.5 rounded-full",
                  micState === "listening" || transcribing
                    ? "bg-primary animate-pulse"
                    : "bg-emerald-500",
                ].join(" ")}
              />
              {micState === "listening"
                ? "Listening"
                : transcribing
                ? "Processing"
                : "Ready"}
            </span>
            <ThemeToggle />
          </div>
        </header>

        {/* ── Conversation ─────────────────────────────────────────────────── */}
        <section className="flex-1 overflow-y-auto -mx-1 px-1">
          <ConversationThread messages={messages} transcribing={transcribing} />
        </section>

        {/* ── Input bar ────────────────────────────────────────────────────── */}
        <div className="flex flex-col gap-3 border-t border-border pt-4">

          {/* Mic + waveform + upload row */}
          <div className="flex items-center gap-3">
            <MicButton state={micState} onToggle={handleMicToggle} />
            <div className="flex-1">
              {micState === "listening" ? (
                <VoiceBar active />
              ) : (
                <p className="text-xs text-muted-foreground">
                  {messages.length === 0
                    ? "Tap the mic to speak"
                    : "Tap to ask something else"}
                </p>
              )}
            </div>
            <AudioUploadButton
              onFile={handleFileUpload}
              disabled={micState === "listening" || transcribing}
            />
          </div>

          {/* Mode + Settings */}
          <div className="flex flex-wrap items-center gap-2">
            <ModeToggle value={mode} onChange={setMode} />
            <div className="h-4 w-px bg-border" aria-hidden="true" />
            <SettingsBar
              settings={settings}
              onChange={(next) => setSettings((s) => ({ ...s, ...next }))}
            />
          </div>
        </div>

      </div>
    </div>
  );
}