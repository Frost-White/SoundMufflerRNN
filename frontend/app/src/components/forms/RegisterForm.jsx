import { useState } from 'react'

const initialValues = { name: '', email: '', password: '', confirmPassword: '' }

export function RegisterForm() {
  const [values, setValues] = useState(initialValues)
  const [message, setMessage] = useState('')
  const [errors, setErrors] = useState({})

  const validate = () => {
    const nextErrors = {}
    if (!values.name.trim()) nextErrors.name = 'Name is required.'
    if (!/\S+@\S+\.\S+/.test(values.email)) nextErrors.email = 'Enter a valid email address.'
    if (values.password.length < 8) nextErrors.password = 'Password must be at least 8 characters.'
    if (values.password !== values.confirmPassword) {
      nextErrors.confirmPassword = 'Passwords do not match.'
    }
    return nextErrors
  }

  const onSubmit = (event) => {
    event.preventDefault()
    const nextErrors = validate()
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return
    setMessage('Mock registration completed.')
    setValues(initialValues)
  }

  return (
    <form className="card form" onSubmit={onSubmit}>
      <h2>Register</h2>
      <Input label="Name" value={values.name} error={errors.name} onChange={(value) => setValues((prev) => ({ ...prev, name: value }))} />
      <Input label="Email" value={values.email} error={errors.email} onChange={(value) => setValues((prev) => ({ ...prev, email: value }))} />
      <Input label="Password" type="password" value={values.password} error={errors.password} onChange={(value) => setValues((prev) => ({ ...prev, password: value }))} />
      <Input
        label="Confirm Password"
        type="password"
        value={values.confirmPassword}
        error={errors.confirmPassword}
        onChange={(value) => setValues((prev) => ({ ...prev, confirmPassword: value }))}
      />
      <button className="btn" type="submit">Sign Up</button>
      {message ? <p className="status status--success">{message}</p> : null}
    </form>
  )
}

function Input({ label, type = 'text', value, onChange, error }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
      {error ? <small className="status status--error">{error}</small> : null}
    </label>
  )
}
