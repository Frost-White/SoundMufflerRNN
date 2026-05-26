const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function enhanceAudio(file, onProgress = () => {}) {
  onProgress(20)

  const formData = new FormData()
  formData.append('file', file)
  onProgress(55)

  const response = await fetch(`${API_BASE_URL}/enhance/web`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error('Audio processing failed.')
  }

  const blob = await response.blob()
  onProgress(100)
  return URL.createObjectURL(blob)
}
