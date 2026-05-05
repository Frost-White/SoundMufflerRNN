import { useCallback, useId, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'

const emailRe = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
const symbolRe = /[^a-zA-Z0-9]/
const lowerRe = /[a-z]/
const upperRe = /[A-Z]/
const digitRe = /\d/
const weakPasswords = new Set([
  'password',
  'password123',
  'qwerty',
  'qwerty123',
  '12345678',
  '123456789',
  'letmein',
  'admin123',
  'welcome123',
])

function hasSequentialRun(text, run = 4) {
  if (text.length < run) return false
  const lower = text.toLowerCase()
  for (let i = 0; i <= lower.length - run; i++) {
    let asc = true
    let desc = true
    for (let j = 1; j < run; j++) {
      const prev = lower.charCodeAt(i + j - 1)
      const curr = lower.charCodeAt(i + j)
      if (curr !== prev + 1) asc = false
      if (curr !== prev - 1) desc = false
    }
    if (asc || desc) return true
  }
  return false
}

function hasRepeatedRun(text, run = 3) {
  if (text.length < run) return false
  let count = 1
  for (let i = 1; i < text.length; i++) {
    if (text[i] === text[i - 1]) {
      count++
      if (count >= run) return true
    } else {
      count = 1
    }
  }
  return false
}

function getPasswordIssue(password, fullName, email) {
  if (!password) return 'Enter a password.'
  if (password.length < 10) return 'Use at least 10 characters.'
  if (password.length > 128) return 'Use 128 characters or fewer.'
  if (!lowerRe.test(password)) return 'Add at least one lowercase letter.'
  if (!upperRe.test(password)) return 'Add at least one uppercase letter.'
  if (!digitRe.test(password)) return 'Add at least one number.'
  if (!symbolRe.test(password)) return 'Add at least one symbol.'

  const normalized = password.toLowerCase()
  if (weakPasswords.has(normalized)) return 'This password is too common.'
  if (hasSequentialRun(password, 4)) return 'Avoid sequential characters like 1234 or abcd.'
  if (hasRepeatedRun(password, 3)) return 'Avoid repeated characters like aaa or 111.'

  const localPart = email.trim().split('@')[0]?.toLowerCase()
  const safeName = fullName.trim().toLowerCase()
  if (localPart && localPart.length >= 3 && normalized.includes(localPart)) {
    return 'Password must not include your email name.'
  }
  if (safeName && safeName.length >= 3) {
    const nameTokens = safeName.split(/\s+/).filter((t) => t.length >= 3)
    if (nameTokens.some((token) => normalized.includes(token))) {
      return 'Password must not include parts of your name.'
    }
  }

  return null
}

function getPasswordStrength(password) {
  if (!password) return 0
  let bits = 0
  if (password.length >= 4) bits++
  if (password.length >= 8) bits++
  if (/[a-z]/.test(password) && /[A-Z]/.test(password)) bits++
  if (/\d/.test(password)) bits++
  if (/[^a-zA-Z0-9]/.test(password)) bits++
  if (bits <= 1) return 1
  if (bits === 2) return 2
  if (bits === 3) return 3
  return 4
}

function CheckIcon() {
  return (
    <svg className="register-form__input-check" width="20" height="20" viewBox="0 0 20 20" aria-hidden>
      <circle cx="10" cy="10" r="10" fill="currentColor" opacity="0.15" />
      <path
        d="M6 10.2l2.4 2.2L14.2 7"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function Spinner() {
  return (
    <span className="register-form__spinner" aria-hidden>
      <span className="register-form__spinner-dot" />
    </span>
  )
}

export function RegisterForm() {
  const { login } = useAuth()
  const baseId = useId()
  const fullNameId = `${baseId}-fullName`
  const emailId = `${baseId}-email`
  const passwordId = `${baseId}-password`
  const confirmId = `${baseId}-confirm`
  const agreeId = `${baseId}-agree`

  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [agreed, setAgreed] = useState(false)
  const [showPassword, setShowPassword] = useState(false)

  const [errors, setErrors] = useState({})
  const [successView, setSuccessView] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [socialHint, setSocialHint] = useState(null)
  const socialTimerRef = useRef(null)

  const strength = getPasswordStrength(password)

  const showNameOk = fullName.trim().length >= 2
  const showEmailOk = emailRe.test(email.trim())
  const passwordIssue = getPasswordIssue(password, fullName, email)
  const showPasswordOk = Boolean(password) && !passwordIssue
  const showConfirmOk =
    confirmPassword.length > 0 && password === confirmPassword && showPasswordOk

  const clearError = useCallback((key) => {
    setErrors((e) => {
      if (!e[key]) return e
      const next = { ...e }
      delete next[key]
      return next
    })
  }, [])

  const resetAll = useCallback(() => {
    setFullName('')
    setEmail('')
    setPassword('')
    setConfirmPassword('')
    setAgreed(false)
    setErrors({})
    setSuccessView(false)
    setSubmitting(false)
  }, [])

  const validate = useCallback(() => {
    const next = {}
    if (!fullName.trim()) next.fullName = 'Enter your full name.'
    if (!email.trim()) next.email = 'Enter your email address.'
    else if (!emailRe.test(email.trim())) next.email = 'Enter a valid email address.'
    const nextPasswordIssue = getPasswordIssue(password, fullName, email)
    if (nextPasswordIssue) next.password = nextPasswordIssue
    if (!confirmPassword) next.confirmPassword = 'Confirm your password.'
    else if (password !== confirmPassword) next.confirmPassword = 'Passwords do not match.'
    if (!agreed) next.agreed = 'You must agree to continue.'
    setErrors(next)
    return Object.keys(next).length === 0
  }, [fullName, email, password, confirmPassword, agreed])

  const onSubmit = (e) => {
    e.preventDefault()
    if (!validate()) return
    setSubmitting(true)
    window.setTimeout(() => {
      setSubmitting(false)
      login({ name: fullName.trim(), email: email.trim() })
      setSuccessView(true)
    }, 1500)
  }

  const onSocialClick = () => {
    if (socialTimerRef.current) window.clearTimeout(socialTimerRef.current)
    setSocialHint(true)
    socialTimerRef.current = window.setTimeout(() => {
      setSocialHint(null)
      socialTimerRef.current = null
    }, 3200)
  }

  if (successView) {
    return (
      <div className="register-form register-form--success" role="status" aria-live="polite">
        <div className="register-form__success-icon" aria-hidden>
          <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
            <circle cx="28" cy="28" r="28" fill="var(--register-success-bg, rgba(111,207,151,0.15))" />
            <path
              d="M16 28.5l8 8 16-16"
              stroke="var(--primary)"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <h1 className="register-form__success-title">You&apos;re all set!</h1>
        <p className="register-form__success-lead">Check your inbox to verify your email.</p>
        <Link className="register-form__btn-primary register-form__btn-primary--wide" to="/login">
          Back to login
        </Link>
      </div>
    )
  }

  const strengthClass =
    strength <= 0
      ? ''
      : strength === 1
        ? 'is-w1'
        : strength === 2
          ? 'is-w2'
          : strength === 3
            ? 'is-w3'
            : 'is-w4'

  return (
    <div className="register-form">
      <div className="register-form__brand">Sound Muffler</div>

      <h1 id="register-main-title" className="register-form__title">
        Create your account
      </h1>
      <p className="register-form__sub">
        Already have an account?{' '}
        <Link className="register-form__inline-link" to="/login">
          Sign in
        </Link>
      </p>

      <form className="register-form__fields" onSubmit={onSubmit} noValidate>
        <div className="register-form__field">
          <label className="register-form__label" htmlFor={fullNameId}>
            Full Name
          </label>
          <div className="register-form__input-row">
            <input
              id={fullNameId}
              name="fullName"
              type="text"
              autoComplete="name"
              className={`register-form__input ${errors.fullName ? 'is-error' : ''} ${showNameOk ? 'is-valid' : ''}`}
              value={fullName}
              onChange={(e) => {
                setFullName(e.target.value)
                clearError('fullName')
              }}
              aria-invalid={!!errors.fullName}
              aria-describedby={errors.fullName ? `${fullNameId}-err` : undefined}
            />
            {showNameOk && !errors.fullName ? <CheckIcon /> : null}
          </div>
          {errors.fullName ? (
            <p id={`${fullNameId}-err`} className="register-form__error" role="alert">
              {errors.fullName}
            </p>
          ) : null}
        </div>

        <div className="register-form__field">
          <label className="register-form__label" htmlFor={emailId}>
            Email Address
          </label>
          <div className="register-form__input-row">
            <input
              id={emailId}
              name="email"
              type="email"
              autoComplete="email"
              className={`register-form__input ${errors.email ? 'is-error' : ''} ${showEmailOk ? 'is-valid' : ''}`}
              value={email}
              onChange={(e) => {
                setEmail(e.target.value)
                clearError('email')
              }}
              aria-invalid={!!errors.email}
              aria-describedby={errors.email ? `${emailId}-err` : undefined}
            />
            {showEmailOk && !errors.email ? <CheckIcon /> : null}
          </div>
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
              autoComplete="new-password"
              className={`register-form__input register-form__input--has-toggle ${errors.password ? 'is-error' : ''} ${showPasswordOk ? 'is-valid' : ''}`}
              value={password}
              onChange={(e) => {
                setPassword(e.target.value)
                clearError('password')
                clearError('confirmPassword')
              }}
              aria-invalid={!!errors.password}
              aria-describedby={[errors.password ? `${passwordId}-err` : null, `${passwordId}-strength`]
                .filter(Boolean)
                .join(' ')}
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
            {showPasswordOk && !errors.password ? <CheckIcon /> : null}
          </div>
          <div
            id={`${passwordId}-strength`}
            className={`register-form__strength ${strengthClass}`}
            role="group"
            aria-label="Password strength"
          >
            {[0, 1, 2, 3].map((i) => (
              <span key={i} className="register-form__strength-seg" data-on={i < strength ? '1' : '0'} />
            ))}
          </div>
          {errors.password ? (
            <p id={`${passwordId}-err`} className="register-form__error" role="alert">
              {errors.password}
            </p>
          ) : null}
        </div>

        <div className="register-form__field">
          <label className="register-form__label" htmlFor={confirmId}>
            Confirm Password
          </label>
          <div className="register-form__input-row">
            <input
              id={confirmId}
              name="confirmPassword"
              type={showPassword ? 'text' : 'password'}
              autoComplete="new-password"
              className={`register-form__input ${errors.confirmPassword ? 'is-error' : ''} ${showConfirmOk ? 'is-valid' : ''}`}
              value={confirmPassword}
              onChange={(e) => {
                setConfirmPassword(e.target.value)
                clearError('confirmPassword')
              }}
              aria-invalid={!!errors.confirmPassword}
              aria-describedby={errors.confirmPassword ? `${confirmId}-err` : undefined}
            />
            {showConfirmOk && !errors.confirmPassword ? <CheckIcon /> : null}
          </div>
          {errors.confirmPassword ? (
            <p id={`${confirmId}-err`} className="register-form__error" role="alert">
              {errors.confirmPassword}
            </p>
          ) : null}
        </div>

        <div className="register-form__field register-form__field--checkbox">
          <div className="register-form__check-row">
            <input
              id={agreeId}
              name="agreed"
              type="checkbox"
              className={errors.agreed ? 'is-error' : ''}
              checked={agreed}
              onChange={(e) => {
                setAgreed(e.target.checked)
                clearError('agreed')
              }}
              aria-invalid={!!errors.agreed}
              aria-describedby={errors.agreed ? `${agreeId}-err` : undefined}
            />
            <label className="register-form__check-label" htmlFor={agreeId}>
              I agree to the{' '}
              <a href="#" className="register-form__inline-link">
                Terms of Service
              </a>{' '}
              and{' '}
              <a href="#" className="register-form__inline-link">
                Privacy Policy
              </a>
            </label>
          </div>
          {errors.agreed ? (
            <p id={`${agreeId}-err`} className="register-form__error" role="alert">
              {errors.agreed}
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
            Create Account
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
            <button
              type="button"
              className="register-form__social-btn"
              onClick={() => onSocialClick()}
            >
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
            <button
              type="button"
              className="register-form__social-btn"
              onClick={() => onSocialClick()}
            >
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
