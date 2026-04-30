import { AccountSettingsForm } from '../components/forms/AccountSettingsForm.jsx'
import { ApiKeyManager } from '../components/forms/ApiKeyManager.jsx'

export function SettingsPage() {
  return (
    <section className="page">
      <AccountSettingsForm />
      <ApiKeyManager />
    </section>
  )
}
