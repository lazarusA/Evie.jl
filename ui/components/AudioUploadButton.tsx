"use client";

import { Paperclip } from "lucide-react";

interface AudioUploadButtonProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function AudioUploadButton({ onFile, disabled }: AudioUploadButtonProps) {
  return (
    <label
      title="Upload an audio or video file"
      className={[
        "flex h-9 w-9 shrink-0 cursor-pointer items-center justify-center rounded-lg",
        "border border-border text-muted-foreground transition-colors",
        disabled
          ? "pointer-events-none opacity-40"
          : "hover:bg-muted hover:text-foreground focus-within:ring-2 focus-within:ring-ring",
      ].join(" ")}
    >
      <Paperclip className="w-4 h-4" />
      <input
        type="file"
        accept="audio/*,video/*"
        className="sr-only"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          // Reset so the same file can be re-selected
          e.target.value = "";
        }}
      />
    </label>
  );
}
