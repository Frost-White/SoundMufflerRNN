export function StatusMessage({ status, progress, error }) {
  if (error) return <p className="status status--error">{error}</p>
  if (status === 'done') return <p className="status status--success">Processing complete.</p>
  if (status === 'loading') {
    return (
      <div>
        <p className="status">Processing... {progress}%</p>
        <progress max={100} value={progress} className="progress" />
      </div>
    )
  }
  return <p className="status">Ready</p>
}
