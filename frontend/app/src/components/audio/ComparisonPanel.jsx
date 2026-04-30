import { WaveformPlayer } from './WaveformPlayer.jsx'

export function ComparisonPanel({ originalUrl, enhancedUrl }) {
  if (!originalUrl && !enhancedUrl) {
    return <p className="status">Upload audio to compare tracks.</p>
  }

  return (
    <div className="comparison-grid">
      <WaveformPlayer audioUrl={originalUrl} title="Original Audio" />
      <WaveformPlayer audioUrl={enhancedUrl} title="Cleaned Audio" />
    </div>
  )
}
