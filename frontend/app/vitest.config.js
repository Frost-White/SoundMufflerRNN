import { defineConfig } from 'vitest/config'

export default defineConfig({
  server: {
    fs: {
      allow: ['../..'],
    },
  },
  test: {
    environment: 'jsdom',
    include: ['../../tests/frontend/**/*.test.js'],
    clearMocks: true,
  },
})
