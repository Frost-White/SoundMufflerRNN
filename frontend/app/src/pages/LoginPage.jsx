import { useLayoutEffect } from 'react'
import { Link } from 'react-router-dom'
import { LoginForm } from '../components/forms/LoginForm.jsx'
import '../styles/register-page.css'

export function LoginPage() {
  useLayoutEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [])

  return (
    <div className="register-viewport">
      <Link className="register-page__back-home" to="/">
        Back to main page
      </Link>
      <div className="register-page">
        <aside className="register-page__left">
          <div className="register-page__left-inner">
            <p className="register-page__heading">Welcome back</p>
            <p className="register-page__lead">
              Pick up where you left off — clean mixes, quieter noise floors, and tools that stay out of
              your way.
            </p>
            <div className="register-page__illustration">
              <svg
                viewBox="0 0 320 200"
                xmlns="http://www.w3.org/2000/svg"
                role="img"
                aria-label="Abstract audio visualization"
              >
                <defs>
                  <linearGradient id="lg1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#6fcf97" stopOpacity="0.9" />
                    <stop offset="100%" stopColor="#3d8b6a" stopOpacity="0.35" />
                  </linearGradient>
                  <linearGradient id="lg2" x1="0%" y1="100%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#6fcf97" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="#9aa4b2" stopOpacity="0.15" />
                  </linearGradient>
                </defs>
                <rect width="320" height="200" rx="12" fill="url(#lg2)" />
                <path
                  d="M20 120 Q80 40 140 100 T260 80 T300 60"
                  fill="none"
                  stroke="url(#lg1)"
                  strokeWidth="6"
                  strokeLinecap="round"
                />
                <circle cx="60" cy="95" r="8" fill="#6fcf97" opacity="0.85" />
                <circle cx="160" cy="72" r="6" fill="#6fcf97" opacity="0.55" />
                <circle cx="240" cy="88" r="10" fill="#6fcf97" opacity="0.7" />
                <rect x="32" y="138" width="12" height="40" rx="4" fill="#6fcf97" opacity="0.45" />
                <rect x="56" y="126" width="12" height="52" rx="4" fill="#6fcf97" opacity="0.65" />
                <rect x="80" y="118" width="12" height="60" rx="4" fill="#6fcf97" opacity="0.55" />
                <rect x="104" y="132" width="12" height="46" rx="4" fill="#6fcf97" opacity="0.75" />
              </svg>
            </div>
          </div>
          <figure className="register-page__testimonial">
            <div className="register-page__avatar" aria-hidden />
            <div className="register-page__quote-body">
              <blockquote className="register-page__quote-text">
                “Do not forget to index your data.”
              </blockquote>
              <figcaption className="register-page__quote-name">Sercan Yücetaş</figcaption>
            </div>
          </figure>
        </aside>

        <section className="register-page__right" aria-label="Log in">
          <div className="register-page__form-shell">
            <LoginForm />
          </div>
        </section>
      </div>
    </div>
  )
}
