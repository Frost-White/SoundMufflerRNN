import { createContext, useCallback, useContext, useMemo, useSyncExternalStore } from 'react'

const STORAGE_KEY = 'soundmuffler_session'

function readSession() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return null
    return { name: String(parsed.name ?? ''), email: String(parsed.email ?? '') }
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
  const user = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)

  const login = useCallback((payload) => {
    setSession({ name: payload.name, email: payload.email })
  }, [])

  const logout = useCallback(() => {
    setSession(null)
  }, [])

  const value = useMemo(
    () => ({
      user,
      isLoggedIn: Boolean(user?.email),
      login,
      logout,
    }),
    [user, login, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
