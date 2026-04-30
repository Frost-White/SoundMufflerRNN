const ALLOWED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/mp3']
const MAX_FILE_BYTES = 20 * 1024 * 1024

export function validateAudioFile(file) {
  if (!file) {
    return 'Please choose a file.'
  }

  if (!ALLOWED_TYPES.includes(file.type)) {
    return 'Supported formats: WAV and MP3.'
  }

  if (file.size > MAX_FILE_BYTES) {
    return 'File size must be under 20MB.'
  }

  return ''
}
