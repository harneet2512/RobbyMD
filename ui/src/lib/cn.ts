import clsx, { type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * `cn` — Tailwind-aware class concatenator. Same convention shadcn/ui uses.
 * Written locally rather than copied from shadcn templates (rules.md §1.1).
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
