import { useCallback, useId, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'
import { apiFetch } from '../../services/api.js'

const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

function Spinner() {
  return (
    <span className="register-form__spinner" aria-hidden>
      <span className="register-form__spinner-dot" />
    </span>
  )
}

export function LoginForm() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const baseId = useId()
  const emailId = `${baseId}-email`
  const passwordId = `${baseId}-password`

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [errors, setErrors] = useState({})
  const [submitting, setSubmitting] = useState(false)

  const clearError = useCallback((key) => {
    setErrors((e) => {
      if (!e[key]) return e
      const next = { ...e }
      delete next[key]
      return next
    })
  }, [])

  const validate = useCallback(() => {
    const next = {}
    if (!email.trim()) next.email = 'Enter your email address.'
    else if (!emailRe.test(email.trim())) next.email = 'Enter a valid email address.'
    if (!password) next.password = 'Enter your password.'
    else if (password.length < 8) next.password = 'Use at least 8 characters.'
    setErrors(next)
    return Object.keys(next).length === 0
  }, [email, password])

  const onSubmit = async (e) => {
    e.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    setErrors((prev) => ({ ...prev, api: undefined }))
    try {
      const trimmed = email.trim()
      const data = await apiFetch('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email: trimmed, password }),
      })
      login({ token: data.access_token, user: data.user })
      navigate('/', { replace: true })
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Login failed'
      setErrors((prev) => ({ ...prev, api: msg }))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="register-form">
      <div className="register-form__brand">Sound Muffler</div>

      <h1 id="login-main-title" className="register-form__title">
        Log in
      </h1>
      <p className="register-form__sub">
        New here?{' '}
        <Link className="register-form__inline-link" to="/register">
          Create an account
        </Link>
      </p>

      <form className="register-form__fields" onSubmit={onSubmit} noValidate>
        <div className="register-form__field">
          <label className="register-form__label" htmlFor={emailId}>
            Email
          </label>
          <input
            id={emailId}
            name="email"
            type="email"
            autoComplete="email"
            className={`register-form__input ${errors.email ? 'is-error' : ''}`}
            value={email}
            onChange={(e) => {
              setEmail(e.target.value)
              clearError('email')
            }}
            aria-invalid={!!errors.email}
            aria-describedby={errors.email ? `${emailId}-err` : undefined}
          />
          {errors.email ? (
            <p id={`${emailId}-err`} className="register-form__error" role="alert">
              {errors.email}
            </p>
          ) : null}
        </div>

        <div className="register-form__field">
          <label className="register-form__label" htmlFor={passwordId}>
            Password
          </label>
          <div className="register-form__input-row register-form__input-row--password">
            <input
              id={passwordId}
              name="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              className={`register-form__input register-form__input--has-toggle ${errors.password ? 'is-error' : ''}`}
              value={password}
              onChange={(e) => {
                setPassword(e.target.value)
                clearError('password')
              }}
              aria-invalid={!!errors.password}
              aria-describedby={errors.password ? `${passwordId}-err` : undefined}
            />
            <button
              type="button"
              className="register-form__eye"
              onClick={() => setShowPassword((v) => !v)}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path
                    d="M3 3l18 18M10.5 10.5a2 2 0 002 2M9.9 5.1A10.4 10.4 0 0112 5c4 0 7.3 2.5 9 6a10.2 10.2 0 01-2.4 3.6M6.4 6.4A9.7 9.7 0 003 11c1.7 3.5 5 6 9 6 .8 0 1.6-.1 2.3-.2"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                  <path
                    d="M7 7a9.8 9.8 0 0010 10"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
              ) : (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path
                    d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  />
                  <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
                </svg>
              )}
            </button>
          </div>
          {errors.password ? (
            <p id={`${passwordId}-err`} className="register-form__error" role="alert">
              {errors.password}
            </p>
          ) : null}
        </div>

        {errors.api ? (
          <p className="register-form__error" role="alert">
            {errors.api}
          </p>
        ) : null}

        <button
          type="submit"
          className="register-form__btn-primary register-form__btn-primary--wide"
          disabled={submitting}
          aria-busy={submitting}
        >
          <span className={submitting ? 'register-form__btn-label is-hidden' : 'register-form__btn-label'}>
            Log in
          </span>
          <span className={submitting ? 'register-form__btn-loading' : 'register-form__btn-loading is-hidden'}>
            <Spinner />
            <span className="visually-hidden">Loading</span>
          </span>
        </button>

        <p className="register-form__footer">
          © 2026 Sound Muffler ·{' '}
          <a href="#" className="register-form__footer-link">
            Privacy
          </a>{' '}
          ·{' '}
          <a href="#" className="register-form__footer-link">
            Terms
          </a>
        </p>
      </form>
    </div>
  )
}
