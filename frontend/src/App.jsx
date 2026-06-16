import { useEffect, useState } from 'react'
import { getDatasetInfo, trainModel } from './api'
import { DEFAULT_CONFIG } from './schema'
import ConfigForm from './ConfigForm'
import Results from './Results'

// One stat shown in the header bar.
function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}

function App() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [result, setResult] = useState(null)
  const [training, setTraining] = useState(false)
  const [trainError, setTrainError] = useState(null)

  useEffect(() => {
    getDatasetInfo()
      .then((data) => {
        setInfo(data)
        const missing = {}
        data.features.forEach((f) => {
          if (f.missing_pct > 0) {
            missing[f.name] = { strategy: 'impute', statistic: f.type === 'numeric' ? 'median' : 'mode' }
          }
        })
        setConfig((current) => ({
          ...current,
          features: data.features.map((f) => f.name),
          missing,
        }))
      })
      .catch((err) => setError(err.message))
  }, [])

  const updateConfig = (field, value) => {
    setConfig((current) => ({ ...current, [field]: value }))
  }

  const handleTrain = async () => {
    setTraining(true)
    setTrainError(null)
    try {
      setResult(await trainModel(config))
    } catch (err) {
      setTrainError(err.message)
    } finally {
      setTraining(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>ML Arena</h1>
          <div className="tagline">Configure → train → read the metrics</div>
        </div>
        {info && (
          <div className="stats">
            <Stat label="Rows" value={info.n_rows.toLocaleString()} />
            <Stat label="Rainy-day rate" value={`${info.positive_rate_pct}%`} />
            <Stat label="Train" value={info.split.train.years} />
            <Stat label="Validation" value={info.split.validation.years} />
            <Stat label="Test" value="hidden" />
          </div>
        )}
      </header>

      {error && <p className="page-error">Could not reach the backend: {error}</p>}
      {!info && !error && <p className="loading">Loading dataset…</p>}

      {info && (
        <div className="layout">
          <div className="config-col">
            <ConfigForm config={config} features={info.features} onChange={updateConfig} />

            <details className="card">
              <summary>Current config (sent to /api/train)</summary>
              <pre>{JSON.stringify(config, null, 2)}</pre>
            </details>
          </div>

          <aside className="results-col">
            <div className="card">
              <button className="train-button" onClick={handleTrain} disabled={training}>
                {training ? 'Training…' : 'Train model'}
              </button>
              {trainError && <div className="train-error">Error: {trainError}</div>}
            </div>

            {result ? (
              <Results result={result} />
            ) : (
              <div className="card placeholder">
                Configure the pipeline, then press <strong>Train model</strong> to see the metric panel here.
              </div>
            )}
          </aside>
        </div>
      )}
    </div>
  )
}

export default App