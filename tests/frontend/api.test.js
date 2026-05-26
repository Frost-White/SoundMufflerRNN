import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiFetch, getStoredToken } from '../../frontend/app/src/services/api'

describe('api helpers', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('reads stored token when present', () => {
    localStorage.setItem('soundmuffler_session', JSON.stringify({ token: 'abc' }))
    expect(getStoredToken()).toBe('abc')
  })

  it('returns null when storage is malformed', () => {
    localStorage.setItem('soundmuffler_session', '{broken')
    expect(getStoredToken()).toBeNull()
  })

  it('adds auth header and parses json body', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify({ ok: true }),
    })

    const out = await apiFetch('/health', { token: 'token-1' })
    expect(out).toEqual({ ok: true })
  })

  it('returns null on 204', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      status: 204,
      text: async () => '',
    })

    const out = await apiFetch('/no-content')
    expect(out).toBeNull()
  })

  it('throws parsed string detail on error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      json: async () => ({ detail: 'Nope' }),
      statusText: 'Bad Request',
    })

    await expect(apiFetch('/x')).rejects.toThrow('Nope')
  })
})
