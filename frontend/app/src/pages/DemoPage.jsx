import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import '../styles/audio-processor-page.css'

const ACCEPT =
  'audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/ogg,.mp3,.wav,.ogg'

const INVALID_FILE_MESSAGE =
  'This file type is not supported. Please use MP3, WAV, or OGG.'

function isAllowedAudioFile(file) {
  const name = file.name.toLowerCase()
  if (/\.(mp3|wav|ogg)$/.test(name)) return true
  const t = (file.type || '').toLowerCase()
  if (!t.startsWith('audio/')) return false
  return (
    t === 'audio/mpeg' ||
    t === 'audio/mp3' ||
    t === 'audio/wav' ||
    t === 'audio/x-wav' ||
    t === 'audio/ogg'
  )
}

function hashString(s) {
  let h = 0
  for (let i = 0; i < s.length; i += 1) {
    h = Math.imul(31, h) + s.charCodeAt(i)
    h |= 0
  }
  return h
}

function mulberry32(seed) {
  return function next() {
    let t = (seed += 0x6d2b79f5)
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function generateBarHeights(seed, count = 72) {
  const rng = mulberry32(seed)
  const raw = Array.from({ length: count }, () => 0.12 + rng() * 0.88)
  return raw.map((_, i, arr) => {
    const prev = arr[i - 1] ?? arr[i]
    const next = arr[i + 1] ?? arr[i]
    const v = (prev + arr[i] * 2 + next) / 4
    return Math.min(1, Math.max(0.08, v))
  })
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '0:00'
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function WaveformStub({ seed, labelId }) {
  const bars = useMemo(() => generateBarHeights(Math.abs(seed), 80), [seed])
  const w = 100
  const h = 40
  const step = w / bars.length
  const gap = step * 0.12

  return (
    <svg
      className="audio-processor__waveform"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      role="img"
      aria-labelledby={labelId}
    >
      <title id={labelId}>Audio waveform placeholder</title>
      {bars.map((height, i) => {
        const barH = height * (h - 2)
        const y = h - barH
        return (
          <rect
            key={i}
            x={i * step + gap / 2}
            y={y}
            width={step - gap}
            height={barH}
            fill="var(--primary)"
            fillOpacity={0.42}
            rx={0.5}
          />
        )
      })}
    </svg>
  )
}

function OriginalPlayer({ src }) {
  const audioRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const a = audioRef.current
    if (!a) return
    setCurrent(0)
    setDuration(0)
    setPlaying(false)
  }, [src])

  const onPlayPause = () => {
    const a = audioRef.current
    if (!a) return
    if (playing) {
      a.pause()
    } else {
      void a.play()
    }
  }

  const onSeek = (e) => {
    const a = audioRef.current
    if (!a || !duration) return
    const t = (Number(e.target.value) / 1000) * duration
    a.currentTime = t
    setCurrent(t)
  }

  const pct = duration ? (current / duration) * 1000 : 0

  return (
    <div className="audio-processor__player">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => {
          setPlaying(false)
          setCurrent(0)
        }}
        onTimeUpdate={() => {
          const a = audioRef.current
          if (a) setCurrent(a.currentTime)
        }}
        onLoadedMetadata={() => {
          const a = audioRef.current
          if (a) setDuration(a.duration || 0)
        }}
      />
      <div className="audio-processor__player-row">
        <button
          type="button"
          className="audio-processor__play-btn"
          onClick={onPlayPause}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden fill="currentColor">
              <rect x="5" y="4" width="5" height="16" rx="1" />
              <rect x="14" y="4" width="5" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden fill="currentColor">
              <path d="M8 5v14l11-7-11-7z" />
            </svg>
          )}
        </button>
        <input
          type="range"
          className="audio-processor__scrub"
          min={0}
          max={1000}
          value={Number.isFinite(pct) ? pct : 0}
          onChange={onSeek}
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={1000}
          aria-valuenow={Math.round(pct)}
        />
        <span className="audio-processor__time" aria-live="polite">
          {formatTime(current)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}

function ProcessedPlayerStub() {
  return (
    <div className="audio-processor__player">
      <div className="audio-processor__player-row">
        <button
          type="button"
          className="audio-processor__play-btn"
          disabled
          aria-label="Play (unavailable)"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden fill="currentColor">
            <path d="M8 5v14l11-7-11-7z" />
          </svg>
        </button>
        <input
          type="range"
          className="audio-processor__scrub"
          min={0}
          max={1000}
          value={0}
          disabled
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={1000}
          aria-valuenow={0}
        />
        <span className="audio-processor__time">0:00 / 0:00</span>
      </div>
      <p className="audio-processor__player-hint">Ready when backend connects</p>
    </div>
  )
}

export function DemoPage() {
  const inputId = 'audio-processor-file'
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [fileError, setFileError] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [isComplete, setIsComplete] = useState(false)

  const objectUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : ''),
    [file],
  )

  useEffect(() => {
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [objectUrl])

  const waveformSeedOriginal = useMemo(
    () =>
      file ? hashString(`${file.name}-${file.size}-original`) : 0,
    [file],
  )
  const waveformSeedProcessed = useMemo(
    () =>
      file ? hashString(`${file.name}-${file.size}-processed`) : 1,
    [file],
  )

  const resetAll = useCallback(() => {
    setFile(null)
    setDragOver(false)
    setFileError('')
    setIsProcessing(false)
    setIsComplete(false)
  }, [])

  const pickFile = (f) => {
    if (!f) return
    if (!isAllowedAudioFile(f)) {
      setFileError(INVALID_FILE_MESSAGE)
      return
    }
    setFileError('')
    setFile(f)
    setIsComplete(false)
    setIsProcessing(false)
  }

  const onInputChange = (e) => {
    const f = e.target.files?.[0]
    pickFile(f)
    e.target.value = ''
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    pickFile(e.dataTransfer.files?.[0])
  }

  const onProcess = () => {
    if (!file || isProcessing) return
    setIsProcessing(true)
    const ms = 2000 + Math.random() * 1000
    window.setTimeout(() => {
      setIsProcessing(false)
      setIsComplete(true)
    }, ms)
  }

  const showControls = Boolean(file)
  const showComparison = Boolean(file && isComplete)

  return (
    <div className="audio-processor">
      <header className="audio-processor__header">
        <span className="audio-processor__brand">Sound Muffler</span>
        <h1 className="audio-processor__title">Audio Processor</h1>
        <p className="audio-processor__tagline">
          Upload your audio, reduce noise, compare and download.
        </p>
      </header>

      <section aria-labelledby={`${inputId}-heading`}>
        <h2 id={`${inputId}-heading`} className="audio-processor__sr-input">
          Upload audio
        </h2>
        <input
          id={inputId}
          type="file"
          accept={ACCEPT}
          className="audio-processor__sr-input"
          aria-label="Choose audio file"
          onChange={onInputChange}
        />

        {!file ? (
          <label
            htmlFor={inputId}
            className="audio-processor__upload-label"
            onDragEnter={(e) => {
              e.preventDefault()
              setFileError('')
              setDragOver(true)
            }}
            onDragOver={(e) => {
              e.preventDefault()
              setDragOver(true)
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget)) {
                setDragOver(false)
              }
            }}
            onDrop={onDrop}
          >
            <div
              className={`audio-processor__dropzone ${dragOver ? 'audio-processor__dropzone--active' : ''}`}
            >
              <svg
                className="audio-processor__drop-icon"
                width="52"
                height="52"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 .75-9.95z" />
                <path d="M12 16V10M9 13l3-3 3 3" />
              </svg>
              <p className="audio-processor__drop-heading">
                Drop your audio file here
              </p>
              <p className="audio-processor__drop-sub">
                or click to browse — supports MP3, WAV, OGG
              </p>
            </div>
          </label>
        ) : null}

        {!file ? (
          fileError ? (
            <p className="audio-processor__error" role="alert">
              {fileError}
            </p>
          ) : null
        ) : null}

        {file ? (
          <div className="audio-processor__file-row">
            <div className="audio-processor__file-meta">
              <span className="audio-processor__file-name">{file.name}</span>
              <span className="audio-processor__file-size">
                {formatFileSize(file.size)}
              </span>
            </div>
            <button
              type="button"
              className="audio-processor__remove-btn"
              onClick={resetAll}
            >
              Remove
            </button>
          </div>
        ) : null}
      </section>

      {showControls ? (
        <section aria-labelledby="process-action-heading">
          <h2 id="process-action-heading" className="audio-processor__sr-input">
            Process audio
          </h2>
          <p className="audio-processor__hint">
            Noise cancellation runs when you process your file.
          </p>
          <div className="audio-processor__process-wrap">
            <button
              type="button"
              className={`audio-processor__process-btn ${isProcessing ? 'audio-processor__process-btn--loading' : ''}`}
              onClick={onProcess}
              disabled={isProcessing || isComplete}
              aria-busy={isProcessing}
            >
              {isProcessing ? (
                <>
                  <span className="audio-processor__spinner" aria-hidden />
                  Processing…
                </>
              ) : isComplete ? (
                'Processing complete'
              ) : (
                'Process Audio'
              )}
            </button>
          </div>
        </section>
      ) : null}

      {showComparison ? (
        <section className="audio-processor__compare" aria-labelledby="compare-heading">
          <div className="audio-processor__compare-top">
            <button type="button" className="audio-processor__start-over" onClick={resetAll}>
              Start over
            </button>
          </div>
          <h2 id="compare-heading" className="audio-processor__sr-input">
            Compare original and processed
          </h2>
          <div className="audio-processor__compare-grid">
            <article className="audio-processor__card">
              <h3 className="audio-processor__card-title">Original</h3>
              <WaveformStub
                seed={waveformSeedOriginal}
                labelId="wf-orig-title"
              />
              <OriginalPlayer src={objectUrl} />
            </article>
            <article className="audio-processor__card">
              <h3 className="audio-processor__card-title">Processed</h3>
              <WaveformStub
                seed={waveformSeedProcessed}
                labelId="wf-proc-title"
              />
              <ProcessedPlayerStub />
            </article>
          </div>
        </section>
      ) : null}

      {showComparison ? (
        <section className="audio-processor__download-block" aria-labelledby="download-heading">
          <h2 id="download-heading" className="audio-processor__sr-input">
            Download
          </h2>
          <button
            type="button"
            className="audio-processor__download-btn"
            disabled
            aria-disabled="true"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
              <path d="M12 4v12m0 0l-4-4m4 4l4-4" />
              <path d="M4 20h16" />
            </svg>
            Download Processed Audio
          </button>
          <p className="audio-processor__download-hint">
            Available once backend is connected.
          </p>
        </section>
      ) : null}
    </div>
  )
}
