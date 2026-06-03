export async function enhanceAudio(file, onProgress = () => {}) {
  onProgress(20)

  const formData = new FormData()
  formData.append('file', file)
  onProgress(55)

  const response = await fetch('/enhance/web', {
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
