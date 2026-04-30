import { useEffect, useMemo, useState } from 'react'
import { ComparisonPanel } from '../components/audio/ComparisonPanel.jsx'
import { UploadZone } from '../components/upload/UploadZone.jsx'

export function DemoPage() {
  const [originalFile, setOriginalFile] = useState(null)
  const [enhancedUrl, setEnhancedUrl] = useState('')
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const originalUrl = useMemo(
    () => (originalFile ? URL.createObjectURL(originalFile) : ''),
    [originalFile],
  )

  useEffect(() => {
    return () => {
      if (originalUrl) URL.revokeObjectURL(originalUrl)
    }
  }, [originalUrl])

  useEffect(() => {
    return () => {
      if (enhancedUrl) URL.revokeObjectURL(enhancedUrl)
    }
  }, [enhancedUrl])

  const handleUpload = async (file) => {
    setOriginalFile(file)
    setError('')
    setStatus('loading')
    setProgress(0)
    try {
      // Backend is not available yet; pass uploaded audio directly.
      setProgress(100)
      setEnhancedUrl(URL.createObjectURL(file))
      setStatus('done')
    } catch (uploadError) {
      setError(uploadError.message || 'An unknown error occurred.')
      setStatus('idle')
    }
  }

  const isUploadOnly = !originalFile

  return (
    <section className={`page demo-page ${isUploadOnly ? 'demo-page--upload-only' : ''}`}>
      <UploadZone
        onFileAccepted={handleUpload}
        disabled={status === 'loading'}
        status={status}
        progress={progress}
        error={error}
        fullscreen={isUploadOnly}
      />
      {originalUrl ? (
        <ComparisonPanel originalUrl={originalUrl} enhancedUrl={enhancedUrl} />
      ) : null}
    </section>
  )
}
