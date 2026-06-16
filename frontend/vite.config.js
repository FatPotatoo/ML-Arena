import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,        // must match the address the backend's CORS allows
    strictPort: true,  // fail loudly if 5173 is taken, instead of silently using another port
  },
})
