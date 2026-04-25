import { useState, useEffect, useRef, useCallback } from "react";

export function useTypewriter(
  text: string,
  active: boolean,
  onComplete?: () => void
) {
  const [charIndex, setCharIndex] = useState(active ? 0 : text.length);
  const rafRef = useRef<number>(0);
  const lastTickRef = useRef(0);
  const completeRef = useRef(onComplete);
  completeRef.current = onComplete;

  const isComplete = charIndex >= text.length;

  useEffect(() => {
    if (!active || isComplete) return;

    const tick = (now: number) => {
      if (now - lastTickRef.current >= 30) {
        lastTickRef.current = now;
        setCharIndex((prev) => {
          const next = prev + 1;
          if (next >= text.length) {
            completeRef.current?.();
          }
          return next;
        });
      }
      if (charIndex < text.length - 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [active, isComplete, text.length, charIndex]);

  const reset = useCallback(() => setCharIndex(0), []);

  return { visibleText: text.slice(0, charIndex), isComplete, reset };
}
