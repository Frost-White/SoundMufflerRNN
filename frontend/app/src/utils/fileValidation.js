const ALLOWED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/mp3']
export const MAX_AUDIO_FILE_BYTES = 2 * 1024 * 1024
const MAX_FILE_BYTES = MAX_AUDIO_FILE_BYTES

export function validateAudioFile(file) {
  if (!file) {
    return 'Please choose a file.'
  }

  if (!ALLOWED_TYPES.includes(file.type)) {
    return 'Supported formats: WAV and MP3.'
  }

  if (file.size > MAX_FILE_BYTES) {
    return 'File size must be 2 MB or smaller.'
  }

  return ''
}
