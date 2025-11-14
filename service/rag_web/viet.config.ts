import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,          // 컨테이너 외부 접근 허용
    port: 5173,
  }
});