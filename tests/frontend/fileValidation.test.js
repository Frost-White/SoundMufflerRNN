import { describe, expect, it } from 'vitest'

import { validateAudioFile } from '../../frontend/app/src/utils/fileValidation'

describe('validateAudioFile', () => {
  it('returns error when no file', () => {
    expect(validateAudioFile(null)).toContain('choose a file')
  })

  it('rejects unsupported type', () => {
    const file = { type: 'text/plain', size: 100 }
    expect(validateAudioFile(file)).toContain('Supported formats')
  })

  it('rejects big files', () => {
    const file = { type: 'audio/wav', size: 21 * 1024 * 1024 }
    expect(validateAudioFile(file)).toContain('under 20MB')
  })

  it('accepts valid files', () => {
    const file = { type: 'audio/mp3', size: 1024 }
    expect(validateAudioFile(file)).toBe('')
  })
})
