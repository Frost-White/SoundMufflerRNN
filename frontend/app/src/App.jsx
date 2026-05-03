import { BrowserRouter, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import { NavBar } from './components/layout/NavBar.jsx'
import { AuthProvider } from './context/AuthContext.jsx'
import { AccountPage } from './pages/AccountPage.jsx'
import { DemoPage } from './pages/DemoPage.jsx'
import { LoginPage } from './pages/LoginPage.jsx'
import { RegisterPage } from './pages/RegisterPage.jsx'

function MainLayout() {
  const { pathname } = useLocation()
  const accountLayout = pathname === '/account'

  return (
    <div className="app-shell">
      <NavBar />
      <main className={`app-content ${accountLayout ? 'app-content--account' : ''}`}>
        <Outlet />
      </main>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/*" element={<MainLayout />}>
            <Route index element={<DemoPage />} />
            <Route path="account" element={<AccountPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}

export default App
