import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite config. Path alias keeps import paths short (src/... -> @/...).
// Server port fixed at 5173 (the Vite default) — Eng_doc.md §9 puts the
// API server on localhost separately, and our MockServer runs in-process
// for Phase 1 so no proxy rules are needed yet.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
