/** @type {import('tailwindcss').Config} */
// Design system — see Eng_doc.md §7 and frontend-design SKILL.
// Direction: editorial / clinical-instrument. One dark field, restrained
// typography, high-contrast status tokens. Not a dashboard. Not SaaS.
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "1.5rem",
    },
    extend: {
      colors: {
        // Semantic tokens (CSS vars in index.css so theming is centralised).
        bg: "hsl(var(--bg))",
        surface: "hsl(var(--surface))",
        "surface-2": "hsl(var(--surface-2))",
        border: "hsl(var(--border))",
        "border-strong": "hsl(var(--border-strong))",
        fg: "hsl(var(--fg))",
        "fg-muted": "hsl(var(--fg-muted))",
        "fg-subtle": "hsl(var(--fg-subtle))",
        // Status tokens — map to clinical states from PRD §6.
        "status-active": "hsl(var(--status-active))",
        "status-superseded": "hsl(var(--status-superseded))",
        "status-confirmed": "hsl(var(--status-confirmed))",
        "status-dismissed": "hsl(var(--status-dismissed))",
        // Branch tokens — four differential branches from PRD §6.3.
        "branch-cardiac": "hsl(var(--branch-cardiac))",
        "branch-pulmonary": "hsl(var(--branch-pulmonary))",
        "branch-msk": "hsl(var(--branch-msk))",
        "branch-gi": "hsl(var(--branch-gi))",
        // Accent (used for provenance highlight; rules.md §4.5).
        accent: "hsl(var(--accent))",
        "accent-soft": "hsl(var(--accent-soft))",
        // Disclaimer header (non-decorative — rules.md §3.7).
        "disclaimer-bg": "hsl(var(--disclaimer-bg))",
        "disclaimer-fg": "hsl(var(--disclaimer-fg))",
        "disclaimer-border": "hsl(var(--disclaimer-border))",
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Inter",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "JetBrains Mono",
          "Menlo",
          "monospace",
        ],
        display: [
          "ui-serif",
          "Georgia",
          "Iowan Old Style",
          "Times New Roman",
          "serif",
        ],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.04em" }],
      },
      letterSpacing: {
        eyebrow: "0.14em",
      },
      boxShadow: {
        panel: "0 1px 0 0 hsl(var(--border) / 0.6), 0 8px 24px -12px hsl(0 0% 0% / 0.45)",
        focus: "0 0 0 2px hsl(var(--accent) / 0.55)",
      },
      transitionTimingFunction: {
        breathe: "cubic-bezier(0.22, 1, 0.36, 1)",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" },
        },
      },
      animation: {
        "fade-in": "fade-in 240ms cubic-bezier(0.22, 1, 0.36, 1) both",
        "pulse-soft": "pulse-soft 1600ms ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
