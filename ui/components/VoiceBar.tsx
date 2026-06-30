"use client";

const BAR_COUNT = 32;
const HEIGHTS = Array.from(
  { length: BAR_COUNT },
  (_, i) => 10 + Math.round(Math.abs(Math.sin(i * 0.72 + 1.1)) * 22)
);

interface VoiceBarProps {
  active: boolean;
}

export function VoiceBar({ active }: VoiceBarProps) {
  if (!active) return null;
  return (
    <div
      aria-hidden="true"
      className="flex items-center gap-[3px] h-8 px-1"
    >
      {HEIGHTS.map((h, i) => (
        <span
          key={i}
          className="w-[3px] rounded-full bg-primary animate-wave"
          style={{
            height: h,
            animationDelay: `${(i * 47) % 900}ms`,
          }}
        />
      ))}
    </div>
  );
}
