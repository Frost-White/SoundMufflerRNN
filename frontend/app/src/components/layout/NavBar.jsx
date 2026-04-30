import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Main Page' },
  { to: '/register', label: 'Register' },
]

export function NavBar() {
  return (
    <header className="nav">
      <h1 className="nav__title">Sound Muffler</h1>
      <nav aria-label="Main menu">
        <ul className="nav__list">
          {links.map((link) => (
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
        </ul>
      </nav>
    </header>
  )
}
