import { useState } from 'react'

export function DemoSamplesPanel({ samples, selectedFile, onSelect }) {
  const [loadingId, setLoadingId] = useState(null)
  const [error, setError] = useState('')

  const onSampleClick = async (sample) => {
    if (loadingId) return
    setLoadingId(sample.id)
    setError('')
    try {
      const res = await fetch(sample.path)
      if (!res.ok) {
        throw new Error('Sample file not found.')
      }
      const blob = await res.blob()
      const filename = sample.filename || `${sample.id}.wav`
      const file = new File([blob], filename, {
        type: blob.type || 'audio/wav',
      })
      onSelect(file)
    } catch {
      setError('Could not load this sample. Add the audio file to public/samples/.')
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <aside className="demo-samples-panel" aria-labelledby="demo-samples-heading">
      <h2 id="demo-samples-heading" className="demo-samples-panel__title">
        Try a sample
      </h2>
      <p className="demo-samples-panel__desc">
        Pick a demo recording, then process it on the right.
      </p>
      <ul className="demo-samples-panel__list">
        {samples.map((sample) => {
          const filename = sample.filename || `${sample.id}.wav`
          const isActive = selectedFile?.name === filename
          const isLoading = loadingId === sample.id
          return (
            <li key={sample.id}>
              <button
                type="button"
                className={`demo-samples-panel__item ${isActive ? 'demo-samples-panel__item--active' : ''}`}
                onClick={() => onSampleClick(sample)}
                disabled={Boolean(loadingId)}
                aria-pressed={isActive}
                aria-busy={isLoading}
              >
                <span className="demo-samples-panel__item-label">{sample.label}</span>
                {isLoading ? (
                  <span className="demo-samples-panel__item-status">Loading…</span>
                ) : isActive ? (
                  <span className="demo-samples-panel__item-status">Selected</span>
                ) : null}
              </button>
            </li>
          )
        })}
      </ul>
      {error ? (
        <p className="demo-samples-panel__error" role="alert">
          {error}
        </p>
      ) : null}
    </aside>
  )
}
