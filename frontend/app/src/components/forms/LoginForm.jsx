import { useCallback, useId, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'

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
  const [socialHint, setSocialHint] = useState(null)
  const socialTimerRef = useRef(null)

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

  const onSubmit = (e) => {
    e.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    window.setTimeout(() => {
      const trimmed = email.trim()
      const displayName = trimmed.includes('@') ? trimmed.split('@')[0] : trimmed || 'User'
      login({ name: displayName.charAt(0).toUpperCase() + displayName.slice(1), email: trimmed })
      setSubmitting(false)
      navigate('/', { replace: true })
    }, 900)
  }

  const onSocialClick = () => {
    if (socialTimerRef.current) window.clearTimeout(socialTimerRef.current)
    setSocialHint(true)
    socialTimerRef.current = window.setTimeout(() => {
      setSocialHint(null)
      socialTimerRef.current = null
    }, 3200)
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

        <div className="register-form__divider">
          <span>or continue with</span>
        </div>

        <div className="register-form__social-wrap">
          <p className="register-form__social-hint" role="status">
            {socialHint ? 'Social login coming soon' : '\u00a0'}
          </p>
          <div className="register-form__social">
            <button type="button" className="register-form__social-btn" onClick={() => onSocialClick()}>
              <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden>
                <path
                  fill="#EA4335"
                  d="M12 10.2v3.7h5.1c-.2 1.1-.9 2.1-1.9 2.7l3 2.3c1.8-1.6 2.8-4 2.8-6.8 0-.7-.1-1.3-.2-1.9H12z"
                />
                <path
                  fill="#34A853"
                  d="M12 22c2.4 0 4.5-.8 6-2.2l-3-2.3c-.8.6-1.9 1-3.1 1-2.4 0-4.4-1.6-5.1-3.8H3.1v2.3C4.6 20 8 22 12 22z"
                />
                <path
                  fill="#4A90E2"
                  d="M6.9 14.7c-.2-.6-.3-1.2-.3-1.7s.1-1.2.3-1.7l-3.1-2.4C2.8 10.5 2.4 11.2 2 12c0 2.2.9 4.2 2.4 5.7l3.5-3z"
                />
                <path
                  fill="#FBBC05"
                  d="M12 5.4c1.3 0 2.5.4 3.4 1.2l2.6-2.6C16.5 2.6 14.4 1.7 12 1.7 8 1.7 4.6 3.7 3.1 7.2l3.5 2.7c.7-2.2 2.7-3.8 5.4-3.8z"
                />
              </svg>
              Google
            </button>
            <button type="button" className="register-form__social-btn" onClick={() => onSocialClick()}>
              <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden>
                <path
                  fill="currentColor"
                  d="M12 .5C5.65.5.5 5.65.5 12c0 5.18 3.36 9.57 8 11.15.58.1.8-.25.8-.56 0-.28 0-1.02 0-2-3.25.7-3.94-1.56-3.94-1.56-.53-1.35-1.3-1.7-1.3-1.7-1.08-.74.08-.72.08-.72 1.2.08 1.84 1.23 1.84 1.23 1.07 1.84 2.8 1.3 3.48 1 .1-.78.42-1.3.76-1.6-2.6-.3-5.33-1.3-5.33-5.8 0-1.28.46-2.33 1.2-3.15-.12-.3-.52-1.52.12-3.18 0 0 1-.32 3.3 1.2a11.5 11.5 0 013-.4c1.02 0 2.04.14 3 .4 2.28-1.52 3.28-1.2 3.28-1.2.64 1.66.24 2.88.12 3.18.76.82 1.2 1.87 1.2 3.15 0 4.52-2.74 5.5-5.36 5.78.42.36.8 1.08.8 2.18 0 1.57-.02 2.84-.02 3.23 0 .31.2.67.8.55A11.48 11.48 0 0023.5 12C23.5 5.65 18.35.5 12 .5z"
                />
              </svg>
              GitHub
            </button>
          </div>
        </div>

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
