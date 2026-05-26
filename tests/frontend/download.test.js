import { describe, expect, it, vi } from 'vitest'

import { downloadFromUrl } from '../../frontend/app/src/utils/download'

describe('downloadFromUrl', () => {
  it('creates a temporary anchor and clicks it', () => {
    const link = document.createElement('a')
    const clickSpy = vi.spyOn(link, 'click').mockImplementation(() => {})
    const removeSpy = vi.spyOn(link, 'remove').mockImplementation(() => {})
    const createSpy = vi.spyOn(document, 'createElement').mockReturnValue(link)
    const appendSpy = vi.spyOn(document.body, 'appendChild')

    downloadFromUrl('blob:test', 'result.wav')

    expect(createSpy).toHaveBeenCalledWith('a')
    expect(link.href).toContain('blob:test')
    expect(link.download).toBe('result.wav')
    expect(appendSpy).toHaveBeenCalledWith(link)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(removeSpy).toHaveBeenCalledTimes(1)
  })
})
