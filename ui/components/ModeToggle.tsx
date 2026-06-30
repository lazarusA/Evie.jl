"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

export type Mode = "fast" | "balanced" | "thorough";

const MODES: { value: Mode; label: string; title: string }[] = [
  { value: "fast",     label: "Fast",     title: "Shortest response, lowest latency" },
  { value: "balanced", label: "Balanced", title: "Default — good quality and speed"  },
  { value: "thorough", label: "Thorough", title: "Longer, more detailed responses"   },
];

interface ModeToggleProps {
  value: Mode;
  onChange: (mode: Mode) => void;
}

export function ModeToggle({ value, onChange }: ModeToggleProps) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={(v) => { if (v) onChange(v as Mode); }}
      aria-label="Response mode"
    >
      {MODES.map((m) => (
        <ToggleGroupItem
          key={m.value}
          value={m.value}
          aria-label={m.label}
          title={m.title}
          className="px-3 text-xs font-medium"
        >
          {m.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}