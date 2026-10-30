import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The dev server forwards /api/* to the FastAPI backend (python -m semcache serve).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    fs: { allow: [".."] }, // data/demo_script.json lives next to frontend/
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
