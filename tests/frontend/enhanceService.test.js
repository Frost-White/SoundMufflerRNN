import { beforeEach, describe, expect, it, vi } from 'vitest'

import { enhanceAudio } from '../../frontend/app/src/services/enhanceService'

describe('enhanceAudio', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('uploads file and reports progress', async () => {
    const blob = new Blob(['x'], { type: 'audio/wav' })
    const file = new File([blob], 'input.wav', { type: 'audio/wav' })
    const onProgress = vi.fn()
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      blob: async () => blob,
    })
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:result')

    const url = await enhanceAudio(file, onProgress)

    expect(url).toBe('blob:result')
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(createObjectURLSpy).toHaveBeenCalledWith(blob)
    expect(onProgress).toHaveBeenCalledWith(20)
    expect(onProgress).toHaveBeenCalledWith(55)
    expect(onProgress).toHaveBeenCalledWith(100)
  })

  it('throws on non-ok response', async () => {
    const file = new File(['a'], 'bad.wav', { type: 'audio/wav' })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
    })

    await expect(enhanceAudio(file)).rejects.toThrow('Audio processing failed.')
  })
})
