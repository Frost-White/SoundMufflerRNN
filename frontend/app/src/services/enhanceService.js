import { getStoredToken } from './api.js'

const SIGN_IN_MESSAGE = 'Please sign in to process audio.'

export async function enhanceAudio(file, onProgress = () => {}) {
  const token = getStoredToken()
  if (!token) {
    throw new Error(SIGN_IN_MESSAGE)
  }

  onProgress(20)

  const formData = new FormData()
  formData.append('file', file)
  onProgress(55)

  const response = await fetch('/enhance/web', {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  })

  if (response.status === 401) {
    throw new Error(SIGN_IN_MESSAGE)
  }

  if (!response.ok) {
    throw new Error('Audio processing failed.')
  }

  const blob = await response.blob()
  onProgress(100)
  return URL.createObjectURL(blob)
}
