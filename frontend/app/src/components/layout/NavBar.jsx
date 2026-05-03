import { NavLink, useLocation } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext.jsx'

const mainLinks = [{ to: '/', label: 'Main Page' }]

export function NavBar() {
  const { isLoggedIn } = useAuth()
  const { pathname } = useLocation()
  const authLinkActive = pathname === '/login' || pathname === '/register'

  return (
    <header className="nav">
      <h1 className="nav__title">Sound Muffler</h1>
      <nav aria-label="Main menu">
        <ul className="nav__list">
          {mainLinks.map((link) => (
            <li key={link.to}>
              <NavLink
                to={link.to}
                className={({ isActive }) =>
                  `nav__link ${isActive ? 'nav__link--active' : ''}`
                }
              >
                {link.label}
              </NavLink>
            </li>
          ))}
          <li>
            {isLoggedIn ? (
              <NavLink
                to="/account"
                className={({ isActive }) =>
                  `nav__link nav__link--account ${isActive ? 'nav__link--active' : ''}`
                }
              >
                ACCOUNT
              </NavLink>
            ) : (
              <NavLink
                to="/login"
                className={() => `nav__link ${authLinkActive ? 'nav__link--active' : ''}`}
              >
                Log in / Register
              </NavLink>
            )}
          </li>
        </ul>
      </nav>
    </header>
  )
}
