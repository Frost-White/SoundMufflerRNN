import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'

export function WaveformPlayer({ audioUrl, title }) {
  const containerRef = useRef(null)
  const waveRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return undefined
    setReady(false)
    setError('')
    const wave = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4f5d79',
      progressColor: '#6fcf97',
      cursorColor: '#f2f2f2',
      height: 70,
      barWidth: 2,
    })
    waveRef.current = wave
    let disposed = false

    const onReady = () => {
      if (disposed) return
      setReady(true)
      setError('')
    }
    const onError = () => {
      if (disposed) return
      setError('Waveform could not be displayed.')
      setReady(false)
    }

    wave.on('ready', onReady)
    wave.on('error', onError)
    wave.on('finish', () => {
      if (disposed) return
      setIsPlaying(false)
    })

    const loadAudio = async () => {
      try {
        if (audioUrl.startsWith('blob:')) {
          const response = await fetch(audioUrl)
          const blob = await response.blob()
          if (disposed) return
          wave.loadBlob(blob)
          return
        }
        wave.load(audioUrl)
      } catch {
        wave.load(audioUrl)
      }
    }
    loadAudio()

    return () => {
      disposed = true
      wave.destroy()
    }
  }, [audioUrl])

  if (!audioUrl) return null

  return (
    <section className="card">
      <h3>{title}</h3>
      {error ? (
        <audio controls src={audioUrl} className="audio-native" />
      ) : (
        <div ref={containerRef} className="waveform" />
      )}
      <div className="audio-actions">
        <button
          type="button"
          className="btn"
          disabled={!ready}
          onClick={() => {
            waveRef.current?.playPause()
            setIsPlaying((prev) => !prev)
          }}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
      </div>
    </section>
  )
}
