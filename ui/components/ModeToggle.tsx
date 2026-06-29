"use client";

export type Mode = "fast" | "balanced" | "thorough";

const MODES: { value: Mode; label: string; title: string }[] = [
  { value: "fast",     label: "Fast",     title: "Shortest response, lowest latency" },
  { value: "balanced", label: "Balanced", title: "Default — good quality and speed" },
  { value: "thorough", label: "Thorough", title: "Longer, more detailed responses" },
];

interface ModeToggleProps {
  value: Mode;
  onChange: (mode: Mode) => void;
}

export function ModeToggle({ value, onChange }: ModeToggleProps) {
  return (
    <div
      role="group"
      aria-label="Response mode"
      className="inline-flex items-center rounded-lg border border-border bg-muted p-0.5 gap-0.5"
    >
      {MODES.map((m) => {
        const active = value === m.value;
        return (
          <button
            key={m.value}
            onClick={() => onChange(m.value)}
            aria-pressed={active}
            title={m.title}
            className={[
              "px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-150",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              active
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            ].join(" ")}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}
