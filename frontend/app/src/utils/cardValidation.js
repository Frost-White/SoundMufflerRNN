export function digitsOnly(s) {
  return s.replace(/\D/g, '')
}

/** @returns {'visa' | 'mastercard' | 'unknown' | null} */
export function detectCardBrand(digits) {
  const d = digitsOnly(digits)
  if (!d) return null
  if (/^4/.test(d)) return 'visa'
  if (/^5[1-5]/.test(d) || /^2[2-7]/.test(d)) return 'mastercard'
  return 'unknown'
}

export function luhnCheck(digits) {
  const d = digitsOnly(digits)
  if (d.length < 13) return false
  let sum = 0
  let alt = false
  for (let i = d.length - 1; i >= 0; i -= 1) {
    let n = parseInt(d[i], 10)
    if (alt) {
      n *= 2
      if (n > 9) n -= 9
    }
    sum += n
    alt = !alt
  }
  return sum % 10 === 0
}

export function expectedCardLength(brand) {
  if (brand === 'visa' || brand === 'mastercard') return 16
  return null
}

export function validateCardNumber(digits) {
  const d = digitsOnly(digits)
  const brand = detectCardBrand(d)
  if (!d.length) return { ok: false, brand: null, code: 'empty' }
  if (brand === 'unknown') return { ok: false, brand, code: 'unsupported' }
  const len = expectedCardLength(brand)
  if (d.length !== len) return { ok: false, brand, code: 'incomplete' }
  if (!luhnCheck(d)) return { ok: false, brand, code: 'luhn' }
  return { ok: true, brand }
}

export function parseExpiryInput(exp) {
  const cleaned = exp.trim().replace(/\s/g, '')
  const slash = cleaned.split('/')
  if (slash.length >= 2) {
    const mm = parseInt(slash[0], 10)
    let yy = parseInt(slash[1], 10)
    if (yy < 100) yy += 2000
    if (mm >= 1 && mm <= 12 && yy >= 2000) return { exp_month: mm, exp_year: yy }
  }
  const compact = cleaned.replace(/\D/g, '')
  if (compact.length >= 4) {
    const mm = parseInt(compact.slice(0, 2), 10)
    let yy = parseInt(compact.slice(2, 4), 10)
    yy += 2000
    if (mm >= 1 && mm <= 12) return { exp_month: mm, exp_year: yy }
  }
  return null
}

export function isExpiryExpired({ exp_month, exp_year }) {
  const now = new Date()
  const y = now.getFullYear()
  const m = now.getMonth() + 1
  if (exp_year < y) return true
  if (exp_year === y && exp_month < m) return true
  return false
}

export function validateExpiry(exp) {
  const parsed = parseExpiryInput(exp)
  if (!parsed) return { ok: false, code: 'invalid' }
  if (isExpiryExpired(parsed)) return { ok: false, code: 'expired', ...parsed }
  return { ok: true, ...parsed }
}

export function validateCvc(cvc) {
  const d = digitsOnly(cvc)
  if (!d.length) return { ok: false, code: 'empty' }
  if (d.length < 3) return { ok: false, code: 'incomplete' }
  if (d.length !== 3) return { ok: false, code: 'invalid' }
  return { ok: true }
}

export function formatCardNumberInput(value) {
  const d = digitsOnly(value).slice(0, 16)
  return d.replace(/(\d{4})(?=\d)/g, '$1 ').trim()
}

export function formatExpiryInput(value) {
  const d = digitsOnly(value).slice(0, 4)
  if (d.length <= 2) return d
  return `${d.slice(0, 2)} / ${d.slice(2)}`
}

export function cardNumberErrorMessage(v) {
  if (v.ok) return null
  if (v.code === 'unsupported') return 'Only Visa and Mastercard are accepted.'
  if (v.code === 'luhn') return 'This card number does not look valid.'
  if (v.code === 'incomplete') {
    const len = expectedCardLength(v.brand)
    return len ? `Enter all ${len} digits.` : 'Enter your card number.'
  }
  return 'Enter your card number.'
}

export function expiryErrorMessage(v) {
  if (v.ok) return null
  if (v.code === 'expired') return 'Expiry date must be this month or later.'
  return 'Enter expiry as MM / YY.'
}

export function cvcErrorMessage(v) {
  if (v.ok) return null
  if (v.code === 'incomplete') return 'Enter the 3-digit security code.'
  return 'Enter a valid CVC.'
}
