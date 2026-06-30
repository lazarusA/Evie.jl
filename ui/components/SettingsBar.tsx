"use client";

import { useState } from "react";
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerDescription,
} from "@/components/ui/drawer";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Check, ChevronDown } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export type DrawerKey = "stt" | "llm" | "device";

interface Option {
  value: string;
  label: string;
  sublabel?: string;
}

export interface AssistantSettings {
  stt: string;
  llm: string;
  device: string;
}

// ── Options ───────────────────────────────────────────────────────────────────

const STT_OPTIONS: Option[] = [
  { value: "whisper-tiny",   label: "Whisper tiny",   sublabel: "74 MB · fastest"        },
  { value: "whisper-base",   label: "Whisper base",   sublabel: "141 MB"                 },
  { value: "whisper-small",  label: "Whisper small",  sublabel: "465 MB · recommended"   },
  { value: "whisper-medium", label: "Whisper medium", sublabel: "1.5 GB"                 },
  { value: "whisper-large",  label: "Whisper large",  sublabel: "2.9 GB · most accurate" },
];

const LLM_OPTIONS: Option[] = [
  { value: "llama-3.2-1b",  label: "Llama 3.2 1B",  sublabel: "~800 MB · very fast"   },
  { value: "llama-3.2-3b",  label: "Llama 3.2 3B",  sublabel: "~2 GB"                 },
  { value: "llama-3.1-8b",  label: "Llama 3.1 8B",  sublabel: "~4.7 GB · recommended" },
  { value: "llama-3.1-70b", label: "Llama 3.1 70B", sublabel: "~40 GB · GPU required"  },
  { value: "mistral-7b",    label: "Mistral 7B",     sublabel: "~4.1 GB"               },
  { value: "phi-3-mini",    label: "Phi-3 Mini",     sublabel: "~2.3 GB · efficient"   },
];

const DEVICE_OPTIONS: Option[] = [
  { value: "cpu",   label: "CPU",          sublabel: "Always available, slower"      },
  { value: "metal", label: "GPU · Metal",  sublabel: "Apple Silicon"                 },
  { value: "cuda",  label: "GPU · CUDA",   sublabel: "NVIDIA — requires CUDA toolkit" },
];

// ── SelectionDrawer ───────────────────────────────────────────────────────────

interface SelectionDrawerProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  options: Option[];
  value: string;
  onSelect: (v: string) => void;
}

function SelectionDrawer({
  open, onClose, title, description, options, value, onSelect,
}: SelectionDrawerProps) {
  return (
    <Drawer open={open} onOpenChange={(o) => !o && onClose()}>
      <DrawerContent className="max-h-[80svh]">
        <DrawerHeader className="text-left px-4 pt-5 pb-3">
          <DrawerTitle>{title}</DrawerTitle>
          {description && (
            <DrawerDescription>{description}</DrawerDescription>
          )}
        </DrawerHeader>

        <Separator />

        <ul className="overflow-y-auto px-3 py-3 pb-8 flex flex-col gap-1">
          {options.map((opt) => {
            const selected = opt.value === value;
            return (
              <li key={opt.value}>
                <Button
                  variant="ghost"
                  className={[
                    "w-full justify-between h-auto py-3 px-3 rounded-xl",
                    selected ? "bg-primary/10 text-primary hover:bg-primary/15 hover:text-primary" : "",
                  ].join(" ")}
                  onClick={() => { onSelect(opt.value); onClose(); }}
                >
                  <span className="flex flex-col items-start gap-0.5">
                    <span className="text-sm font-medium">{opt.label}</span>
                    {opt.sublabel && (
                      <span className="text-xs text-muted-foreground font-normal">
                        {opt.sublabel}
                      </span>
                    )}
                  </span>
                  {selected
                    ? <Check className="w-4 h-4 shrink-0" />
                    : <Badge variant="outline" className="text-[10px] font-normal shrink-0">
                        select
                      </Badge>
                  }
                </Button>
              </li>
            );
          })}
        </ul>
      </DrawerContent>
    </Drawer>
  );
}

// ── SettingPill ───────────────────────────────────────────────────────────────

function SettingPill({
  label, value, onClick,
}: {
  label: string;
  value: string;
  onClick: () => void;
}) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className="h-8 gap-1.5 text-xs font-normal"
    >
      <span className="text-muted-foreground">{label}</span>
      <span className="font-semibold text-foreground truncate max-w-[6rem]">{value}</span>
      <ChevronDown className="w-3 h-3 text-muted-foreground" />
    </Button>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────

interface SettingsBarProps {
  settings: AssistantSettings;
  onChange: (next: Partial<AssistantSettings>) => void;
}

export function SettingsBar({ settings, onChange }: SettingsBarProps) {
  const [open, setOpen] = useState<DrawerKey | null>(null);

  const sttLabel = STT_OPTIONS.find((o) => o.value === settings.stt)?.label  ?? settings.stt;
  const llmLabel = LLM_OPTIONS.find((o) => o.value === settings.llm)?.label  ?? settings.llm;
  const devLabel = DEVICE_OPTIONS.find((o) => o.value === settings.device)?.label ?? settings.device;

  return (
    <>
      <div className="flex items-center gap-1.5 flex-wrap">
        <SettingPill label="STT"    value={sttLabel} onClick={() => setOpen("stt")}    />
        <SettingPill label="LLM"    value={llmLabel} onClick={() => setOpen("llm")}    />
        <SettingPill label="Device" value={devLabel} onClick={() => setOpen("device")} />
      </div>

      <SelectionDrawer
        open={open === "stt"}
        onClose={() => setOpen(null)}
        title="Speech-to-text model"
        description="Converts your voice to text before the LLM sees it. Larger is more accurate."
        options={STT_OPTIONS}
        value={settings.stt}
        onSelect={(v) => onChange({ stt: v })}
      />
      <SelectionDrawer
        open={open === "llm"}
        onClose={() => setOpen(null)}
        title="Language model"
        description="The LLM that generates Evie's responses. Runs fully offline."
        options={LLM_OPTIONS}
        value={settings.llm}
        onSelect={(v) => onChange({ llm: v })}
      />
      <SelectionDrawer
        open={open === "device"}
        onClose={() => setOpen(null)}
        title="Compute device"
        description="GPU acceleration significantly reduces response latency."
        options={DEVICE_OPTIONS}
        value={settings.device}
        onSelect={(v) => onChange({ device: v })}
      />
    </>
  );
}