import { defineConfig } from "vite";

// Relative asset paths, so dist/ works from any directory or host.
export default defineConfig({
  base: "./",
  build: { target: "es2022", assetsInlineLimit: 0, chunkSizeWarningLimit: 1200 },
});
