import { useEffect, useState } from 'react'
import { getDatasetInfo } from './api'
import { DEFAULT_CONFIG } from './schema'
import ConfigForm from './ConfigForm'

function App() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  // The single source of truth for all the knobs. ConfigForm displays it and
  // asks to change it; App owns it.
  const [config, setConfig] = useState(DEFAULT_CONFIG)

  useEffect(() => {
    getDatasetInfo()
      .then((data) => {
        setInfo(data)
        // Default: every feature checked (matches the backend's "all" default).
        setConfig((current) => ({ ...current, features: data.features.map((f) => f.name) }))
      })
      .catch((err) => setError(err.message))
  }, [])

  // Update one field immutably: copy the old config, overwrite one key.
  // (React only re-renders when you give it a NEW object, not a mutated one.)
  const updateConfig = (field, value) => {
    setConfig((current) => ({ ...current, [field]: value }))
  }

  return (
    <main>
      <h1>ML Arena</h1>
      <p>Configure a model, train it, and read the metrics — no code required.</p>

      {error && <p style={{ color: 'crimson' }}>Could not reach the backend: {error}</p>}
      {!info && !error && <p>Loading dataset…</p>}

      {info && (
        <>
          <section className="card">
            <h2>{info.dataset}</h2>
            <ul>
              <li>Target: <code>{info.target}</code> · {info.n_rows.toLocaleString()} rows · rainy-day rate {info.positive_rate_pct}%</li>
              <li>Train: {info.split.train.rows.toLocaleString()} ({info.split.train.years}) · Validation: {info.split.validation.rows.toLocaleString()} ({info.split.validation.years}) · Test: hidden</li>
            </ul>
          </section>

          <ConfigForm config={config} features={info.features} onChange={updateConfig} />

          {/* Temporary: shows the config object updating live as you click.
              In Step 4 this is replaced by a "Train" button + the results. */}
          <details className="card">
            <summary>Current config (this is what we'll send to /api/train)</summary>
            <pre>{JSON.stringify(config, null, 2)}</pre>
          </details>
        </>
      )}
    </main>
  )
}

export default App
