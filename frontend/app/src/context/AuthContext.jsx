import { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore } from 'react'
import { apiFetch } from '../services/api.js'

const STORAGE_KEY = 'soundmuffler_session'

function normalizeUser(u) {
  if (!u || typeof u !== 'object') return null
  const fullName = String(u.full_name ?? u.name ?? '')
  return {
    id: u.id,
    email: String(u.email ?? ''),
    full_name: fullName,
    name: fullName,
    email_verified: Boolean(u.email_verified),
  }
}

function readSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return null
    if (!parsed.token) return null
    const user = normalizeUser(parsed.user)
    if (!user?.email) return null
    return { token: String(parsed.token), user }
  } catch {
    return null
  }
}

let session = readSession()
const listeners = new Set()

function subscribe(cb) {
  listeners.add(cb)
  return () => listeners.delete(cb)
}

function getSnapshot() {
  return session
}

function setSession(next) {
  session = next
  try {
    if (next) localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    else localStorage.removeItem(STORAGE_KEY)
  } catch {
    /* ignore */
  }
  listeners.forEach((l) => l())
}

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const auth = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)

  useEffect(() => {
    let cancelled = false
    const s = readSession()
    if (!s?.token) return undefined

    apiFetch('/auth/me', { token: s.token })
      .then((me) => {
        if (!cancelled) setSession({ token: s.token, user: normalizeUser(me) })
      })
      .catch(() => {
        if (!cancelled) setSession(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback((payload) => {
    const user = normalizeUser(payload.user)
    if (!user || !payload.token) return
    setSession({ token: String(payload.token), user })
  }, [])

  const logout = useCallback(() => {
    setSession(null)
  }, [])

  const value = useMemo(
    () => ({
      user: auth?.user ?? null,
      token: auth?.token ?? null,
      isLoggedIn: Boolean(auth?.token && auth?.user?.email),
      login,
      logout,
    }),
    [auth, login, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
