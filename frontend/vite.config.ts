// frontend/vite.config.ts
import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/main.ts"),
      name: "HometownHubMap",
      fileName: "hometown-hub-map",
      formats: ["iife"]
    },
    rollupOptions: {
      external: ["deck.gl", "google.maps"],
      output: {
        globals: {
          "deck.gl": "deck",
          "google.maps": "google"
        }
      }
    }
  },
  server: {
    port: 5173
  }
});