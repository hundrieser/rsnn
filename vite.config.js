import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  base: '/rsnn/',
  plugins: [react()],
  build: {
    outDir: 'docs',
    emptyOutDir: true,
  },
  resolve: {
    alias: {
      'react-router-dom': fileURLToPath(new URL('./src/lib/hash-router.js', import.meta.url)),
    },
  },
})
