import React, { useState, useEffect, useRef } from 'react';
import { Text } from 'ink';
import { SPINNER } from '../config.js';
import { SPINNER_VERBS } from '../data/spinnerVerbs.js';

interface SpinnerProps {
  /** Token count to display alongside elapsed time. */
  tokens?: number;
}

/** Format elapsed seconds into a human-readable duration string. */
function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${String(s).padStart(2, '0')}s`;
}

/** Abbreviate large token counts with a 'k' suffix. */
function formatTokens(count: number): string {
  if (count >= 1000) return `${(count / 1000).toFixed(1)}k`;
  return String(count);
}

/** Pick a random Korean proverb from the verb list. */
function randomVerb(): string {
  return SPINNER_VERBS[Math.floor(Math.random() * SPINNER_VERBS.length)];
}

export function Spinner({ tokens = 0 }: SpinnerProps) {
  const [frame, setFrame] = useState(0);
  const [verb, setVerb] = useState(randomVerb);
  // Record the mount time so elapsed seconds are accurate even after re-renders
  const startTime = useRef(Date.now());

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame((f) => {
        const next = f + 1;
        // Rotate the displayed proverb every verbChangeInterval frames
        if (next > 0 && next % SPINNER.verbChangeInterval === 0) {
          setVerb(randomVerb());
        }
        return next;
      });
    }, 1000 / SPINNER.fps);
    return () => clearInterval(interval);
  }, []);

  const char = SPINNER.chars[frame % SPINNER.chars.length];
  const elapsed = Math.floor((Date.now() - startTime.current) / 1000);
  const elapsedStr = formatElapsed(elapsed);

  // Append token count only when the backend has reported usage
  const suffix =
    tokens > 0
      ? ` (${elapsedStr} · ↓ ${formatTokens(tokens)} tokens)`
      : ` (${elapsedStr})`;

  return (
    <Text dimColor>
      {char} {verb}…{suffix}
    </Text>
  );
}
