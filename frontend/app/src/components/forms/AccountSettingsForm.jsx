import { useMemo, useState } from 'react'

const initialForm = { name: 'Demo User', email: 'demo@example.com', password: '', confirmPassword: '' }

export function AccountSettingsForm() {
  const [values, setValues] = useState(initialForm)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  const dirty = useMemo(
    () =>
      values.name !== initialForm.name ||
      values.email !== initialForm.email ||
      values.password.length > 0 ||
      values.confirmPassword.length > 0,
    [values],
  )

  const handleSave = async (event) => {
    event.preventDefault()
    if (values.password && values.password !== values.confirmPassword) {
      setMessage('Sifreler eslesmiyor.')
      return
    }
    setSaving(true)
    setMessage('')
    await new Promise((resolve) => setTimeout(resolve, 400))
    setSaving(false)
    setMessage('Ayarlar mock olarak kaydedildi.')
  }

  return (
    <form className="card form" onSubmit={handleSave}>
      <h2>Profil Ayarlari</h2>
      <Input label="Ad" value={values.name} onChange={(value) => setValues((prev) => ({ ...prev, name: value }))} />
      <Input label="E-posta" value={values.email} onChange={(value) => setValues((prev) => ({ ...prev, email: value }))} />
      <Input label="Yeni Sifre" type="password" value={values.password} onChange={(value) => setValues((prev) => ({ ...prev, password: value }))} />
      <Input
        label="Yeni Sifre Tekrar"
        type="password"
        value={values.confirmPassword}
        onChange={(value) => setValues((prev) => ({ ...prev, confirmPassword: value }))}
      />
      <div className="row">
        <button className="btn" type="submit" disabled={!dirty || saving}>
          {saving ? 'Kaydediliyor...' : 'Kaydet'}
        </button>
        <button className="btn btn--ghost" type="button" onClick={() => setValues(initialForm)} disabled={saving || !dirty}>
          Iptal
        </button>
      </div>
      {message ? <p className="status">{message}</p> : null}
    </form>
  )
}

function Input({ label, type = 'text', value, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}
