'use client';

import { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';
import { FiSun, FiMoon } from 'react-icons/fi';

type ThemeToggleProps = {
  className?: string;
};

export function ThemeToggle({ className }: ThemeToggleProps) {
  const [mounted, setMounted] = useState(false);
  const { resolvedTheme, setTheme } = useTheme();

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <Button
      variant="ghost"
      size="icon"
      className={`rounded-md p-2 hover:bg-accent/50 ${className ?? ''}`}
      onClick={() => setTheme(resolvedTheme === 'light' ? 'dark' : 'light')}
      title={`Switch to ${resolvedTheme === 'light' ? 'dark' : 'light'} theme`}
    >
      {resolvedTheme === 'light' ? (
        <FiMoon className="size-5" />
      ) : (
        <FiSun className="size-5" />
      )}
    </Button>
  );
}