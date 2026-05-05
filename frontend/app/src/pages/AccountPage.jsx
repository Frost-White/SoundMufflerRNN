import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'
import '../styles/account-page.css'

function useFocusTrap(containerRef, active) {
  useEffect(() => {
    if (!active) return
    const el = containerRef.current
    if (!el) return
    const prev = document.activeElement
    const getFocusable = () =>
      [
        ...el.querySelectorAll(
          'button:not([disabled]), [href]:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ].filter((n) => !n.hasAttribute('disabled'))

    const onKey = (e) => {
      if (e.key !== 'Tab') return
      const focusables = getFocusable()
      if (focusables.length === 0) return
      const first = focusables[0]
      const last = focusables[focusables.length - 1]
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault()
          last.focus()
        }
      } else if (document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    el.addEventListener('keydown', onKey)
    queueMicrotask(() => getFocusable()[0]?.focus())
    return () => {
      el.removeEventListener('keydown', onKey)
      if (prev && typeof prev.focus === 'function') prev.focus()
    }
  }, [active, containerRef])
}

function IconAccount({ className }) {
  return (
    <svg className={className} width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 11a4 4 0 100-8 4 4 0 000 8zm-7 9a7 7 0 0114 0v1H5v-1z"
        fill="currentColor"
      />
    </svg>
  )
}

function IconBilling({ className }) {
  return (
    <svg className={className} width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 6a2 2 0 012-2h12a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm2 0v2h16V6M6 12h4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  )
}

function IconKey({ className }) {
  return (
    <svg className={className} width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M15.5 14.5L19 11l-3-3-3.5 3.5M9 17l2.5-2.5M7 7a4 4 0 108 0 4 4 0 00-8 0z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function IconLock() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M7 11V8a5 5 0 0110 0v3M6 11h12v10H6V11z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function IconWarning() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M12 8v5M12 17h.01" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <path
        d="M4.5 19h15L12 5 4.5 19z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function IconMenu() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  )
}

function IconCopy() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M9 9V5a2 2 0 012-2h8a2 2 0 012 2v10a2 2 0 01-2 2h-4M9 9H7a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-2"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

function IconCheck() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path d="M5 12l4 4L19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

function randomApiKey() {
  const bytes = new Uint8Array(16)
  crypto.getRandomValues(bytes)
  const hex = [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('')
  return `sk-live-${hex}`
}

function maskFromFull(full) {
  const tail = full.slice(-4)
  return `sk-••••••••••••••••••••${tail}`
}

function newKeyRowId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `k-${crypto.randomUUID()}`
  }
  return `k-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`
}

const NAV = [
  { id: 'account', label: 'Account', icon: IconAccount },
  { id: 'billing', label: 'Billing', icon: IconBilling },
  { id: 'api', label: 'API Keys', icon: IconKey },
]

const PLAN_OPTIONS = [
  { id: 'free', name: 'Free', price: '$0/mo', blurb: 'For individuals and light use.', current: true },
  { id: 'pro', name: 'Pro', price: '$10/mo', blurb: 'Unlimited projects and API access.' },
  { id: 'enterprise', name: 'Enterprise', price: 'Custom', blurb: 'SLA, SSO, and dedicated support.' },
]

export function AccountPage() {
  const { isLoggedIn, user, logout } = useAuth()
  const navigate = useNavigate()
  const baseId = useId()

  const scrollRef = useRef(null)
  const accountRef = useRef(null)
  const billingRef = useRef(null)
  const apiRef = useRef(null)

  const addCardModalRef = useRef(null)
  const revealModalRef = useRef(null)
  const keyNameModalRef = useRef(null)
  const planModalRef = useRef(null)

  const [drawerOpen, setDrawerOpen] = useState(false)
  const [activeNav, setActiveNav] = useState('account')

  const [paymentMethods, setPaymentMethods] = useState([
    { id: 'pm1', brand: 'visa', last4: '4242', expiry: '12/26', isDefault: true },
    { id: 'pm2', brand: 'mastercard', last4: '8888', expiry: '08/25', isDefault: false },
  ])
  const [removePaymentId, setRemovePaymentId] = useState(null)
  const paymentCardRefs = useRef({})
  const pendingSwapRef = useRef(null)

  const [addCardOpen, setAddCardOpen] = useState(false)
  const [addCardLoading, setAddCardLoading] = useState(false)
  const [addCardSuccess, setAddCardSuccess] = useState(false)
  const [cardNumber, setCardNumber] = useState('')
  const [cardExpiry, setCardExpiry] = useState('')
  const [cardCvc, setCardCvc] = useState('')

  const [apiKeys, setApiKeys] = useState([
    { id: 'k1', name: 'Production Key', created: 'Jan 15, 2024', masked: 'sk-••••••••••••••••••••3f9a' },
    { id: 'k2', name: 'Test Key', created: 'Mar 2, 2024', masked: 'sk-••••••••••••••••••••8b2c' },
  ])
  const [revokeKeyId, setRevokeKeyId] = useState(null)
  const [removingKeyId, setRemovingKeyId] = useState(null)
  const [copyFlashId, setCopyFlashId] = useState(null)
  const copyTimerRef = useRef(null)

  const [generateLoading, setGenerateLoading] = useState(false)
  const [revealOpen, setRevealOpen] = useState(false)
  const [revealedKey, setRevealedKey] = useState('')
  const [pendingMaskedRow, setPendingMaskedRow] = useState(null)

  const [keyNameModalOpen, setKeyNameModalOpen] = useState(false)
  const [newKeyNameInput, setNewKeyNameInput] = useState('')
  const [keyNameError, setKeyNameError] = useState(null)

  const [planModalOpen, setPlanModalOpen] = useState(false)
  const [planChangeLoading, setPlanChangeLoading] = useState(false)
  const [planChangeDone, setPlanChangeDone] = useState(false)
  const [cancelSubConfirm, setCancelSubConfirm] = useState(false)
  const [cancelSubNote, setCancelSubNote] = useState(false)

  useFocusTrap(addCardModalRef, addCardOpen)
  useFocusTrap(revealModalRef, revealOpen)
  useFocusTrap(keyNameModalRef, keyNameModalOpen)
  useFocusTrap(planModalRef, planModalOpen)

  const scrollToSection = useCallback((id) => {
    const map = { account: accountRef, billing: billingRef, api: apiRef }
    setActiveNav(id)
    map[id]?.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setDrawerOpen(false)
  }, [])

  const closeAddCard = useCallback(() => {
    setAddCardOpen(false)
    setAddCardLoading(false)
    setAddCardSuccess(false)
    setCardNumber('')
    setCardExpiry('')
    setCardCvc('')
  }, [])

  const finishReveal = useCallback((saveKey) => {
    setRevealOpen(false)
    setRevealedKey('')
    setPendingMaskedRow((pending) => {
      if (saveKey && pending) {
        setApiKeys((list) =>
          list.some((k) => k.id === pending.id) ? list : [pending, ...list],
        )
      }
      return null
    })
  }, [])

  const closeKeyNameModal = useCallback(() => {
    setKeyNameModalOpen(false)
    setNewKeyNameInput('')
    setKeyNameError(null)
  }, [])

  const openKeyNameModal = useCallback(() => {
    setNewKeyNameInput('')
    setKeyNameError(null)
    setKeyNameModalOpen(true)
  }, [])

  const closePlanModal = useCallback(() => {
    setPlanModalOpen(false)
    setPlanChangeLoading(false)
    setPlanChangeDone(false)
  }, [])

  const openPlanModal = useCallback(() => {
    setPlanChangeLoading(false)
    setPlanChangeDone(false)
    setPlanModalOpen(true)
  }, [])

  const stubPlanChange = useCallback(() => {
    setPlanChangeLoading(true)
    setPlanChangeDone(false)
    window.setTimeout(() => {
      setPlanChangeLoading(false)
      setPlanChangeDone(true)
    }, 700)
  }, [])

  const runGenerateFlow = useCallback((displayName) => {
    setGenerateLoading(true)
    window.setTimeout(() => {
      const key = randomApiKey()
      setRevealedKey(key)
      setPendingMaskedRow({
        id: newKeyRowId(),
        name: displayName,
        created: new Date().toLocaleDateString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
        }),
        masked: maskFromFull(key),
      })
      setGenerateLoading(false)
      setRevealOpen(true)
    }, 1000)
  }, [])

  const confirmGenerateFromModal = () => {
    const name = newKeyNameInput.trim()
    if (!name) {
      setKeyNameError('Enter a key name.')
      return
    }
    if (name.length > 80) {
      setKeyNameError('Use 80 characters or fewer.')
      return
    }
    setKeyNameModalOpen(false)
    setNewKeyNameInput('')
    setKeyNameError(null)
    runGenerateFlow(name)
  }

  useEffect(() => {
    let obs = null
    const frame = window.requestAnimationFrame(() => {
      const root = scrollRef.current
      if (!root) return
      const els = [accountRef.current, billingRef.current, apiRef.current].filter(Boolean)
      if (els.length === 0) return

      obs = new IntersectionObserver(
        (entries) => {
          const hit = entries
            .filter((e) => e.isIntersecting && e.intersectionRatio >= 0.25)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
          if (hit?.target?.id === 'account-section') setActiveNav('account')
          if (hit?.target?.id === 'billing-section') setActiveNav('billing')
          if (hit?.target?.id === 'api-section') setActiveNav('api')
        },
        { root, threshold: [0.2, 0.35, 0.5], rootMargin: '-10% 0px -45% 0px' },
      )
      els.forEach((el) => obs.observe(el))
    })
    return () => {
      window.cancelAnimationFrame(frame)
      obs?.disconnect()
    }
  }, [])

  useEffect(() => {
    if (!addCardOpen && !revealOpen && !keyNameModalOpen && !planModalOpen) return
    const onEsc = (e) => {
      if (e.key !== 'Escape') return
      if (revealOpen) {
        finishReveal(false)
        return
      }
      if (keyNameModalOpen) {
        closeKeyNameModal()
        return
      }
      if (planModalOpen) {
        closePlanModal()
        return
      }
      closeAddCard()
    }
    window.addEventListener('keydown', onEsc)
    return () => window.removeEventListener('keydown', onEsc)
  }, [
    addCardOpen,
    revealOpen,
    keyNameModalOpen,
    planModalOpen,
    closeAddCard,
    closeKeyNameModal,
    closePlanModal,
    finishReveal,
  ])

  useLayoutEffect(() => {
    const pendingSwap = pendingSwapRef.current
    if (!pendingSwap) return
    pendingSwapRef.current = null
    pendingSwap.ids.forEach((cardId) => {
      const beforeTop = pendingSwap.beforeTops[cardId]
      const el = paymentCardRefs.current[cardId]
      if (!el || typeof beforeTop !== 'number') return
      const deltaY = beforeTop - el.getBoundingClientRect().top
      if (Math.abs(deltaY) < 1) return
      el.animate(
        [{ transform: `translateY(${deltaY}px)` }, { transform: 'translateY(0)' }],
        { duration: 360, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
      )
    })
  }, [paymentMethods])

  const onSaveCardStub = () => {
    setAddCardLoading(true)
    window.setTimeout(() => {
      setAddCardLoading(false)
      setAddCardSuccess(true)
    }, 900)
  }

  const confirmRemovePayment = (id) => {
    setPaymentMethods((list) => {
      const next = list.filter((p) => p.id !== id)
      if (next.length > 0 && !next.some((p) => p.isDefault)) {
        return next.map((p, i) => ({ ...p, isDefault: i === 0 }))
      }
      return next
    })
    setRemovePaymentId(null)
  }

  const makeDefaultPayment = (id) => {
    setPaymentMethods((list) => {
      const previousDefaultId = list.find((p) => p.isDefault)?.id
      if (!previousDefaultId || previousDefaultId === id) return list
      pendingSwapRef.current = {
        ids: [previousDefaultId, id],
        beforeTops: {
          [previousDefaultId]: paymentCardRefs.current[previousDefaultId]?.getBoundingClientRect().top,
          [id]: paymentCardRefs.current[id]?.getBoundingClientRect().top,
        },
      }
      return list.map((p) => ({ ...p, isDefault: p.id === id }))
    })
  }

  const copyKey = (text, rowId) => {
    navigator.clipboard?.writeText(text).catch(() => {})
    if (copyTimerRef.current) window.clearTimeout(copyTimerRef.current)
    setCopyFlashId(rowId)
    copyTimerRef.current = window.setTimeout(() => {
      setCopyFlashId(null)
      copyTimerRef.current = null
    }, 2000)
  }

  const startRevoke = (id) => {
    setRevokeKeyId(id)
  }

  const confirmRevoke = (id) => {
    setRemovingKeyId(id)
    window.setTimeout(() => {
      setApiKeys((list) => list.filter((k) => k.id !== id))
      setRevokeKeyId(null)
      setRemovingKeyId(null)
    }, 320)
  }

  const onSignOut = () => {
    logout()
    navigate('/')
  }

  if (!isLoggedIn) {
    return <Navigate to="/login" replace />
  }

  const displayName = user?.name?.trim() || 'Demo User'
  const displayEmail = user?.email?.trim() || 'you@example.com'
  const renewsOn = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })

  const NavButtons = ({ classPrefix = 'account-page' }) => (
    <>
      {NAV.map((item) => {
        const Icon = item.icon
        return (
          <button
            key={item.id}
            type="button"
            className={`${classPrefix}__nav-btn ${activeNav === item.id ? 'is-active' : ''}`}
            onClick={() => scrollToSection(item.id)}
          >
            <Icon className={`${classPrefix}__nav-icon`} />
            {item.label}
          </button>
        )
      })}
    </>
  )

  return (
    <div className="account-page">
      <aside className="account-page__sidebar" aria-label="Account sections">
        <p className="account-page__brand">Sound Muffler</p>
        <nav className="account-page__nav" aria-label="Account navigation">
          <NavButtons />
        </nav>
        <div className="account-page__user">
          <p className="account-page__user-name">{displayName}</p>
          <p className="account-page__user-email">{displayEmail}</p>
          <button type="button" className="account-page__sign-out" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      </aside>

      <div
        id="account-mobile-drawer"
        className={`account-mobile-drawer ${drawerOpen ? 'is-open' : ''}`}
        aria-hidden={!drawerOpen}
      >
        <div
          className="account-mobile-drawer__backdrop"
          onClick={() => setDrawerOpen(false)}
          aria-hidden
        />
        <div className="account-mobile-drawer__panel" role="dialog" aria-modal="true" aria-label="Menu">
          <p className="account-page__brand">Sound Muffler</p>
          <nav className="account-page__nav" aria-label="Account navigation">
            <NavButtons />
          </nav>
          <div className="account-page__user">
            <p className="account-page__user-name">{displayName}</p>
            <p className="account-page__user-email">{displayEmail}</p>
            <button type="button" className="account-page__sign-out" onClick={onSignOut}>
              Sign out
            </button>
          </div>
        </div>
      </div>

      <div className="account-page__main">
        <div className="account-page__mobile-bar">
          <button
            type="button"
            className="account-page__menu-toggle"
            aria-expanded={drawerOpen}
            aria-controls="account-mobile-drawer"
            onClick={() => setDrawerOpen((o) => !o)}
          >
            <IconMenu />
            <span className="sr-only">Open menu</span>
          </button>
          <h2 className="account-page__mobile-title">Account</h2>
        </div>

        <div className="account-page__tabs" role="tablist" aria-label="Account sections">
          {NAV.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={activeNav === item.id}
              className={`account-page__tab ${activeNav === item.id ? 'is-active' : ''}`}
              onClick={() => scrollToSection(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="account-page__scroll" ref={scrollRef}>
          <div className="account-page__inner">
            <section
              ref={accountRef}
              id="account-section"
              className="account-page__section"
              aria-labelledby={`${baseId}-account-h`}
            >
              <h1 id={`${baseId}-account-h`} className="account-page__h1">
                Account
              </h1>
              <p className="account-page__lead">
                Your profile information set during registration. Contact support to make changes.
              </p>

              <div className="account-readonly-card">
                <div className="account-readonly-row">
                  <span className="account-readonly-row__label">Full Name</span>
                  <div className="account-readonly-row__input-wrap">
                    <input
                      id={`${baseId}-ro-name`}
                      className="account-readonly-row__input"
                      readOnly
                      aria-readonly="true"
                      value={displayName}
                    />
                  </div>
                  <span className="account-readonly-row__badge">
                    <IconLock /> Read only
                  </span>
                </div>
                <div className="account-readonly-row">
                  <span className="account-readonly-row__label">Email Address</span>
                  <div className="account-readonly-row__input-wrap">
                    <input
                      id={`${baseId}-ro-email`}
                      className="account-readonly-row__input"
                      readOnly
                      aria-readonly="true"
                      value={displayEmail}
                    />
                  </div>
                  <span className="account-readonly-row__badge">
                    <IconLock /> Read only
                  </span>
                </div>
              </div>
              <p className="account-page__note">
                Name and email cannot be changed. If you need to update them please contact support.
              </p>

              <h2 id={`${baseId}-plan-mgmt-h`} className="account-page__h2">
                Plan management
              </h2>
              <p className="account-page__lead account-page__lead--compact">
                View your subscription and switch plans. Changes apply after checkout when billing is connected.
              </p>

              <div className="account-plan-mgmt" aria-labelledby={`${baseId}-plan-mgmt-h`}>
                <div className="account-plan-mgmt__hero">
                  <div>
                    <p className="account-plan-mgmt__eyebrow">Current plan</p>
                    <p className="account-plan-mgmt__title">Free</p>
                  </div>
                  <span className="account-badge">Active</span>
                </div>
                <dl className="account-plan-mgmt__dl">
                  <div>
                    <dt>Billing cycle</dt>
                    <dd>Monthly</dd>
                  </div>
                  <div>
                    <dt>Price</dt>
                    <dd>$0 USD / month</dd>
                  </div>
                  <div>
                    <dt>Renews on</dt>
                    <dd>{renewsOn}</dd>
                  </div>
                </dl>
                <div className="account-plan-mgmt__actions">
                  <button type="button" className="account-btn account-btn--primary" onClick={openPlanModal}>
                    Change plan
                  </button>
                  {cancelSubConfirm ? (
                    <div className="account-plan-mgmt__confirm" role="status" aria-live="polite">
                      <span className="account-plan-mgmt__confirm-text">End subscription after this period?</span>
                      <button
                        type="button"
                        className="account-btn account-btn--danger"
                        onClick={() => {
                          setCancelSubConfirm(false)
                          setCancelSubNote(true)
                        }}
                      >
                        Yes
                      </button>
                      <button
                        type="button"
                        className="account-btn account-btn--ghost"
                        onClick={() => setCancelSubConfirm(false)}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      className="account-btn account-btn--ghost"
                      onClick={() => setCancelSubConfirm(true)}
                      disabled={cancelSubNote}
                    >
                      Cancel subscription
                    </button>
                  )}
                </div>
                {cancelSubNote && (
                  <p className="account-page__stub-note" role="status" aria-live="polite">
                    Cancellation scheduled — available once backend is connected.
                  </p>
                )}
              </div>
            </section>

            <section
              ref={billingRef}
              id="billing-section"
              className="account-page__section"
              aria-labelledby={`${baseId}-billing-h`}
            >
              <h1 id={`${baseId}-billing-h`} className="account-page__h1">
                Billing
              </h1>
              <p className="account-page__lead">
                Manage cards on file. Your plan is managed under Account → Plan management.
              </p>

              <h2 className="account-page__h2">Payment methods</h2>
              <ul className="account-pay-list">
                {[...paymentMethods]
                  .sort((a, b) => Number(b.isDefault) - Number(a.isDefault))
                  .map((pm) => (
                  <li
                    key={pm.id}
                    className="account-pay-card"
                    ref={(el) => {
                      if (el) paymentCardRefs.current[pm.id] = el
                      else delete paymentCardRefs.current[pm.id]
                    }}
                  >
                    <div className="account-pay-card__brand">
                      <span
                        className={`brand-badge brand-badge--${pm.brand}`}
                        aria-label={pm.brand === 'visa' ? 'Visa' : 'Mastercard'}
                      >
                        {pm.brand === 'visa' ? 'VISA' : 'MC'}
                      </span>
                    </div>
                    <div className="account-pay-card__meta">
                      <div className="account-pay-card__digits">•••• {pm.last4}</div>
                      <div className="account-pay-card__exp">Expires {pm.expiry}</div>
                    </div>
                    <div className="account-pay-card__actions">
                      {removePaymentId === pm.id ? (
                        <div
                          className="account-pay-card__confirm"
                          role="status"
                          aria-live="polite"
                        >
                          <span>Are you sure?</span>
                          <button
                            type="button"
                            className="account-btn account-btn--danger"
                            onClick={() => confirmRemovePayment(pm.id)}
                          >
                            Yes
                          </button>
                          <button
                            type="button"
                            className="account-btn account-btn--ghost"
                            onClick={() => setRemovePaymentId(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <>
                          {pm.isDefault ? (
                            <span className="account-pay-card__default">Default</span>
                          ) : (
                            <button
                              type="button"
                              className="account-btn account-btn--link"
                              onClick={() => makeDefaultPayment(pm.id)}
                              aria-label={`Set card ending in ${pm.last4} as default payment method`}
                            >
                              Make default
                            </button>
                          )}
                          <button
                            type="button"
                            className="account-btn account-btn--ghost"
                            onClick={() => setRemovePaymentId(pm.id)}
                          >
                            Remove
                          </button>
                        </>
                      )}
                    </div>
                  </li>
                ))}
              </ul>

              <button
                type="button"
                className="account-btn account-btn--dashed"
                onClick={() => {
                  setAddCardSuccess(false)
                  setAddCardOpen(true)
                }}
              >
                Add payment method
              </button>
            </section>

            <section
              ref={apiRef}
              id="api-section"
              className="account-page__section"
              aria-labelledby={`${baseId}-api-h`}
            >
              <h1 id={`${baseId}-api-h`} className="account-page__h1">
                API Keys
              </h1>
              <p className="account-page__lead">
                Use these keys to authenticate requests to the API. Keep them secret.
              </p>

              <div className="account-warn-banner">
                <IconWarning />
                <p className="account-warn-banner__text">
                  Never share your API keys. Do not expose them in client-side code or public repositories.
                </p>
              </div>

              <div>
                {apiKeys.map((row) => (
                  <div
                    key={row.id}
                    className={`account-api-row ${removingKeyId === row.id ? 'account-api-row--leaving' : ''}`}
                  >
                    <div>
                      <p className="account-api-row__title">{row.name}</p>
                      <p className="account-api-row__date">Created {row.created}</p>
                    </div>
                    <div className="account-api-row__key-wrap">
                      <code className="account-api-row__key">{row.masked}</code>
                      <div className="account-copy-wrap">
                        <button
                          type="button"
                          className="account-icon-btn"
                          aria-label="Copy API key"
                          onClick={() => copyKey(row.masked, row.id)}
                        >
                          {copyFlashId === row.id ? <IconCheck /> : <IconCopy />}
                        </button>
                        {copyFlashId === row.id && (
                          <span className="account-tooltip" role="status">
                            Copied
                          </span>
                        )}
                      </div>
                    </div>
                    <div style={{ justifySelf: 'end' }}>
                      {revokeKeyId === row.id ? (
                        <div role="status" aria-live="polite" className="account-pay-card__confirm">
                          <span>Revoke this key?</span>
                          <button
                            type="button"
                            className="account-btn account-btn--danger"
                            onClick={() => confirmRevoke(row.id)}
                          >
                            Yes
                          </button>
                          <button
                            type="button"
                            className="account-btn account-btn--ghost"
                            onClick={() => setRevokeKeyId(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          className="account-btn account-btn--ghost"
                          onClick={() => startRevoke(row.id)}
                        >
                          Revoke
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <button
                type="button"
                className="account-btn account-btn--primary account-btn--min"
                style={{ marginTop: '0.75rem' }}
                disabled={generateLoading}
                onClick={openKeyNameModal}
              >
                {generateLoading ? (
                  <>
                    <span className="account-btn__spinner" aria-hidden />
                    <span className="sr-only">Loading</span>
                  </>
                ) : (
                  'Generate new API key'
                )}
              </button>
            </section>
          </div>
        </div>
      </div>

      {keyNameModalOpen && (
        <>
          <div className="account-backdrop" aria-hidden onClick={closeKeyNameModal} />
          <div
            className="account-modal"
            ref={keyNameModalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${baseId}-key-name-title`}
          >
            <h2 id={`${baseId}-key-name-title`} className="account-modal__title">
              Name your API key
            </h2>
            <form
              className="account-modal__field"
              onSubmit={(e) => {
                e.preventDefault()
                confirmGenerateFromModal()
              }}
            >
              <label className="account-modal__label" htmlFor={`${baseId}-key-name`}>
                Key name
              </label>
              <input
                id={`${baseId}-key-name`}
                className="account-modal__input"
                autoComplete="off"
                placeholder="e.g. Production"
                value={newKeyNameInput}
                onChange={(e) => {
                  setNewKeyNameInput(e.target.value)
                  if (keyNameError) setKeyNameError(null)
                }}
                aria-invalid={keyNameError ? 'true' : undefined}
                aria-describedby={keyNameError ? `${baseId}-key-name-err` : undefined}
              />
              {keyNameError && (
                <p id={`${baseId}-key-name-err`} className="status status--error" style={{ fontSize: '0.8rem', marginTop: '0.35rem' }} role="alert">
                  {keyNameError}
                </p>
              )}
              <div className="account-modal__actions">
                <button type="button" className="account-btn account-btn--ghost" onClick={closeKeyNameModal}>
                  Cancel
                </button>
                <button type="submit" className="account-btn account-btn--primary account-btn--min">
                  Continue
                </button>
              </div>
            </form>
          </div>
        </>
      )}

      {planModalOpen && (
        <>
          <div className="account-backdrop" aria-hidden onClick={closePlanModal} />
          <div
            className="account-modal"
            ref={planModalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${baseId}-plan-modal-title`}
          >
            <h2 id={`${baseId}-plan-modal-title`} className="account-modal__title">
              Change plan
            </h2>
            <p className="account-page__lead account-page__lead--compact">Pick a plan to continue. Checkout will open when billing is live.</p>
            <div className="account-plan-picker">
              {PLAN_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  className={`account-plan-picker__tier ${opt.current ? 'is-current' : ''}`}
                  disabled={planChangeLoading}
                  onClick={stubPlanChange}
                >
                  <div className="account-plan-picker__tier-head">
                    <span className="account-plan-picker__tier-name">{opt.name}</span>
                    <span className="account-plan-picker__tier-meta">
                      {opt.current && <span className="account-badge">Current</span>}
                      <span className="account-plan-picker__tier-price">{opt.price}</span>
                    </span>
                  </div>
                  <p className="account-plan-picker__tier-blurb">{opt.blurb}</p>
                </button>
              ))}
            </div>
            {planChangeLoading && (
              <p className="account-plan-picker__status" role="status" aria-live="polite">
                <span className="account-plan-picker__status-inner">
                  <span className="account-btn__spinner account-btn__spinner--on-ghost" aria-hidden />
                  Applying…
                </span>
              </p>
            )}
            {planChangeDone && (
              <p className="account-page__stub-note" role="status" aria-live="polite">
                Plan update queued — available once backend is connected.
              </p>
            )}
            <div className="account-modal__actions">
              <button type="button" className="account-btn account-btn--ghost" onClick={closePlanModal}>
                Close
              </button>
            </div>
          </div>
        </>
      )}

      {addCardOpen && (
        <>
          <div className="account-backdrop" aria-hidden onClick={closeAddCard} />
          <div
            className="account-modal"
            ref={addCardModalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${baseId}-add-card-title`}
          >
            <h2 id={`${baseId}-add-card-title`} className="account-modal__title">
              Add payment method
            </h2>
            <div className="account-modal__field">
              <label className="account-modal__label" htmlFor={`${baseId}-cn`}>
                Card number
              </label>
              <input
                id={`${baseId}-cn`}
                className="account-modal__input"
                inputMode="numeric"
                autoComplete="cc-number"
                placeholder="4242 4242 4242 4242"
                value={cardNumber}
                onChange={(e) => setCardNumber(e.target.value)}
              />
            </div>
            <div className="account-modal__field" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.65rem' }}>
              <div>
                <label className="account-modal__label" htmlFor={`${baseId}-exp`}>
                  Expiry
                </label>
                <input
                  id={`${baseId}-exp`}
                  className="account-modal__input"
                  autoComplete="cc-exp"
                  placeholder="MM / YY"
                  value={cardExpiry}
                  onChange={(e) => setCardExpiry(e.target.value)}
                />
              </div>
              <div>
                <label className="account-modal__label" htmlFor={`${baseId}-cvc`}>
                  CVC
                </label>
                <input
                  id={`${baseId}-cvc`}
                  className="account-modal__input"
                  autoComplete="cc-csc"
                  placeholder="123"
                  value={cardCvc}
                  onChange={(e) => setCardCvc(e.target.value)}
                />
              </div>
            </div>
            <div className="account-modal__actions">
              <button type="button" className="account-btn account-btn--ghost" onClick={closeAddCard}>
                Cancel
              </button>
              <button
                type="button"
                className="account-btn account-btn--primary account-btn--min"
                disabled={addCardLoading || addCardSuccess}
                onClick={onSaveCardStub}
              >
                {addCardLoading ? (
                  <>
                    <span className="account-btn__spinner" aria-hidden />
                    <span className="sr-only">Saving</span>
                  </>
                ) : (
                  'Save card'
                )}
              </button>
            </div>
            {addCardSuccess && (
              <p className="account-page__stub-note" role="status" aria-live="polite">
                Card saved — available once backend is connected.
              </p>
            )}
          </div>
        </>
      )}

      {revealOpen && (
        <>
          <div className="account-backdrop" aria-hidden onClick={() => finishReveal(false)} />
          <div
            className="account-modal"
            ref={revealModalRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby={`${baseId}-reveal-title`}
          >
            <h2 id={`${baseId}-reveal-title`} className="account-modal__title">
              Your new API key
            </h2>
            <div className="account-reveal-box" tabIndex={0}>
              {revealedKey}
            </div>
            <button
              type="button"
              className="account-btn account-btn--primary"
              style={{ width: '100%' }}
              aria-label="Copy full API key to clipboard"
              onClick={() => copyKey(revealedKey, 'reveal')}
            >
              {copyFlashId === 'reveal' ? (
                <>
                  <span aria-hidden>
                    <IconCheck />
                  </span>
                  <span className="sr-only">Copied</span>
                </>
              ) : (
                <>
                  <span aria-hidden>
                    <IconCopy />
                  </span>
                  Copy key
                </>
              )}
            </button>
            <p className="account-reveal-warn">Copy this key now. You will not be able to see it again.</p>
            <button
              type="button"
              className="account-btn account-btn--ghost"
              style={{ width: '100%' }}
              onClick={() => finishReveal(true)}
            >
              Done
            </button>
            <p className="account-page__stub-note" style={{ marginTop: '0.75rem' }}>
              Available once backend is connected.
            </p>
          </div>
        </>
      )}
    </div>
  )
}
