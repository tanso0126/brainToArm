import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  root: resolve(__dirname, "windows"),
  publicDir: resolve(__dirname, "public"),
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "../windows_release/assets/ui"),
    emptyOutDir: true,
  },
  server: {
    fs: { allow: [resolve(__dirname)] },
  },
});
