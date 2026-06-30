"use client";

import { Mic, Square } from "lucide-react";

export type MicState = "idle" | "listening";

interface MicButtonProps {
  state: MicState;
  onToggle: () => void;
}

// Animated ring shown while listening
function PulseRing() {
  return (
    <span
      aria-hidden="true"
      className="absolute inset-0 rounded-full bg-primary/20 animate-ping"
    />
  );
}

export function MicButton({ state, onToggle }: MicButtonProps) {
  const listening = state === "listening";
  return (
    <button
      onClick={onToggle}
      aria-label={listening ? "Stop listening" : "Talk to Evie"}
      className={[
        "relative flex h-14 w-14 shrink-0 items-center justify-center rounded-full transition-colors duration-200",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        listening
          ? "bg-primary text-primary-foreground"
          : "bg-muted text-muted-foreground hover:bg-muted/70 hover:text-foreground border border-border",
      ].join(" ")}
    >
      {listening && <PulseRing />}
      {listening
        ? <Square className="w-5 h-5 fill-current relative z-10" />
        : <Mic className="w-5 h-5 relative z-10" />
      }
    </button>
  );
}
