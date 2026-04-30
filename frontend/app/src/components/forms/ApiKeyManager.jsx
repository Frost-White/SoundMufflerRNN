import { useEffect, useState } from 'react'
import { createKey, listKeys, revokeKey } from '../../services/mockApiKeyService.js'

export function ApiKeyManager() {
  const [keys, setKeys] = useState([])
  const [loading, setLoading] = useState(false)
  const [visibleKeyId, setVisibleKeyId] = useState('')

  useEffect(() => {
    listKeys().then(setKeys)
  }, [])

  const handleCreate = async () => {
    setLoading(true)
    const newKey = await createKey()
    setKeys((prev) => [newKey, ...prev])
    setLoading(false)
  }

  const handleRevoke = async (id) => {
    const confirmed = window.confirm('Bu API key iptal edilsin mi?')
    if (!confirmed) return
    setLoading(true)
    await revokeKey(id)
    setKeys((prev) => prev.filter((item) => item.id !== id))
    setLoading(false)
  }

  return (
    <section className="card">
      <h3>API Key Yonetimi</h3>
      <button className="btn" type="button" onClick={handleCreate} disabled={loading}>
        Yeni Key Uret
      </button>
      <ul className="key-list">
        {keys.map((key) => (
          <li key={key.id} className="key-list__item">
            <strong>{key.label}</strong>
            <code>{visibleKeyId === key.id ? key.value : `${key.value.slice(0, 8)}...`}</code>
            <small>{new Date(key.createdAt).toLocaleString()}</small>
            <div className="row">
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setVisibleKeyId((prev) => (prev === key.id ? '' : key.id))}
              >
                {visibleKeyId === key.id ? 'Gizle' : 'Goster'}
              </button>
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => navigator.clipboard.writeText(key.value)}
              >
                Kopyala
              </button>
              <button
                type="button"
                className="btn btn--danger"
                onClick={() => handleRevoke(key.id)}
                disabled={loading}
              >
                Iptal Et
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}
