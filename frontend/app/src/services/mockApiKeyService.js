let keys = [
  { id: crypto.randomUUID(), label: 'Default Key', value: 'sm_1234567890abcdef', createdAt: new Date().toISOString() },
]

export async function listKeys() {
  await wait(100)
  return keys
}

export async function createKey() {
  await wait(200)
  const newKey = {
    id: crypto.randomUUID(),
    label: `Key ${keys.length + 1}`,
    value: `sm_${crypto.randomUUID().replaceAll('-', '')}`,
    createdAt: new Date().toISOString(),
  }
  keys = [newKey, ...keys]
  return newKey
}

export async function revokeKey(id) {
  await wait(150)
  keys = keys.filter((key) => key.id !== id)
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
