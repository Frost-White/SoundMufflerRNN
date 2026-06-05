import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  // Docker Compose exposes backend on :8000. Override with VITE_API_PROXY if needed.
  const apiTarget = env.VITE_API_PROXY || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
        },
        '/enhance': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
