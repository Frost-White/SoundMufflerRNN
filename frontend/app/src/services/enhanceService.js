const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export async function enhanceAudio(file, onProgress) {
  onProgress(20)

  const shouldMock = !import.meta.env.VITE_API_BASE_URL
  if (shouldMock) {
    return mockEnhance(file, onProgress)
  }

  const formData = new FormData()
  formData.append('file', file)
  onProgress(55)

  const response = await fetch(`${API_BASE_URL}/enhance`, {
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

async function mockEnhance(file, onProgress) {
  await delay(400)
  onProgress(45)
  await delay(400)
  onProgress(80)
  await delay(300)
  onProgress(100)
  return URL.createObjectURL(file)
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
