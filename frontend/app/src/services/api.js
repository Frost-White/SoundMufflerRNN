const API_ROOT = '/api'

function apiUrl(path) {
  const p = path.startsWith('/') ? path : `/${path}`
  return `${API_ROOT}${p}`
}

async function parseError(res) {
  const err = await res.json().catch(() => ({}))
  const d = err.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return d.map((x) => x.msg ?? JSON.stringify(x)).join(', ')
  if (d && typeof d === 'object' && 'message' in d) return String(d.message)
  return res.statusText || 'Request failed'
}

export function getStoredToken() {
  try {
    const raw = localStorage.getItem('soundmuffler_session')
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed?.token ? String(parsed.token) : null
  } catch {
    return null
  }
}

export async function apiFetch(path, options = {}) {
  const token = options.token ?? getStoredToken()
  const headers = { ...options.headers }
  const isFormData = options.body instanceof FormData
  if (!isFormData && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json'
  }
  if (token) headers.Authorization = `Bearer ${token}`
  const res = await fetch(apiUrl(path), { ...options, headers })
  if (!res.ok) throw new Error(await parseError(res))
  if (res.status === 204) return null
  const text = await res.text()
  if (!text) return null
  return JSON.parse(text)
}
