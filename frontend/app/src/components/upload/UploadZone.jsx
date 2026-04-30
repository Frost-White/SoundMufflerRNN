import { useRef, useState } from 'react'
import { validateAudioFile } from '../../utils/fileValidation.js'
import { StatusMessage } from '../common/StatusMessage.jsx'

export function UploadZone({ onFileAccepted, disabled, status, progress, error, fullscreen }) {
  const inputRef = useRef(null)
  const [validationError, setValidationError] = useState('')
  const [isDragOver, setIsDragOver] = useState(false)

  const processFile = (file) => {
    const validationError = validateAudioFile(file)
    setValidationError(validationError)
    if (!validationError) onFileAccepted(file)
  }

  return (
    <section className={`card ${fullscreen ? 'upload-card--fullscreen' : ''}`}>
      <h2>Upload Audio</h2>
      <div
        className={`upload-zone ${isDragOver ? 'upload-zone--active' : ''} ${fullscreen ? 'upload-zone--fullscreen' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(event) => {
          event.preventDefault()
          setIsDragOver(false)
          processFile(event.dataTransfer.files[0])
        }}
      >
        <p>Drag and drop your file here, or choose a file.</p>
        <button
          type="button"
          className="btn"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          Choose File
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,audio/wav,audio/mpeg"
          hidden
          onChange={(event) => processFile(event.target.files?.[0])}
        />
      </div>
      {validationError ? <p className="status status--error">{validationError}</p> : null}
      <StatusMessage status={status} progress={progress} error={error} />
    </section>
  )
}
