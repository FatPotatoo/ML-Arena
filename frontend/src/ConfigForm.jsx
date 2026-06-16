import {
  OPTIONS,
  MISSING_STRATEGY,
  STATISTIC_NUMERIC,
  STATISTIC_TEXT,
  defaultStatistic,
} from './schema'

// A small reusable dropdown for the simple (global) knobs.
function Select({ label, field, config, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={config[field]} onChange={(e) => onChange(field, e.target.value)}>
        {OPTIONS[field].map(([value, text]) => (
          <option key={value} value={value}>{text}</option>
        ))}
      </select>
    </label>
  )
}

export default function ConfigForm({ config, features, onChange }) {
  const toggleFeature = (name) => {
    const next = config.features.includes(name)
      ? config.features.filter((f) => f !== name)
      : [...config.features, name]
    onChange('features', next)
  }

  const missingFeatures = features.filter(
    (f) => f.missing_pct > 0 && config.features.includes(f.name),
  )

  const setMissing = (name, patch) => {
    const type = features.find((f) => f.name === name)?.type
    const current = config.missing[name] || { strategy: 'impute', statistic: defaultStatistic(type) }
    onChange('missing', { ...config.missing, [name]: { ...current, ...patch } })
  }

  const setAllStrategy = (strategy) => {
    const next = { ...config.missing }
    missingFeatures.forEach((f) => {
      const current = next[f.name] || { statistic: defaultStatistic(f.type) }
      next[f.name] = { strategy, statistic: current.statistic || defaultStatistic(f.type) }
    })
    onChange('missing', next)
  }

  return (
    <>
      <section className="card">
        <h3 className="card-title">Preprocessing</h3>
        <div className="grid">
          <Select label="Location encoding" field="location_encoding" config={config} onChange={onChange} />
          <Select label="Wind encoding" field="wind_encoding" config={config} onChange={onChange} />
          <Select label="Scaling" field="scaling" config={config} onChange={onChange} />
        </div>
      </section>

      <section className="card">
        <h3 className="card-title">Missing values (per feature)</h3>
        {missingFeatures.length === 0 ? (
          <p><small>No selected feature has missing values.</small></p>
        ) : (
          <>
            <div className="toolbar">
              <span>Set all to:</span>
              <button type="button" onClick={() => setAllStrategy('impute')}>Impute</button>
              <button type="button" onClick={() => setAllStrategy('drop_row')}>Drop rows</button>
              <button type="button" onClick={() => setAllStrategy('drop_col')}>Exclude</button>
            </div>
            <div className="missing-list">
              {missingFeatures.map((f) => {
                const entry = config.missing[f.name] || { strategy: 'impute', statistic: defaultStatistic(f.type) }
                const statisticOptions = f.type === 'numeric' ? STATISTIC_NUMERIC : STATISTIC_TEXT
                return (
                  <div key={f.name} className="missing-row">
                    <span className="missing-name">{f.name} <small>({f.missing_pct}%)</small></span>
                    <select value={entry.strategy} onChange={(e) => setMissing(f.name, { strategy: e.target.value })}>
                      {MISSING_STRATEGY.map(([value, text]) => (
                        <option key={value} value={value}>{text}</option>
                      ))}
                    </select>
                    {entry.strategy === 'impute' && (
                      <select value={entry.statistic} onChange={(e) => setMissing(f.name, { statistic: e.target.value })}>
                        {statisticOptions.map(([value, text]) => (
                          <option key={value} value={value}>{text}</option>
                        ))}
                      </select>
                    )}
                  </div>
                )
              })}
            </div>
          </>
        )}
      </section>

      <section className="card">
        <h3 className="card-title">Model</h3>
        <div className="grid">
          <Select label="Class weight" field="class_weight" config={config} onChange={onChange} />
          <Select label="Leakage policy" field="leakage_policy" config={config} onChange={onChange} />
          <Select label="Penalty" field="penalty" config={config} onChange={onChange} />
        </div>
        <div className="grid" style={{ marginTop: '0.85rem' }}>
          <label className="field">
            <span>Regularization C: {config.C}</span>
            <input
              type="range" min="-3" max="3" step="0.1"
              value={Math.log10(config.C)}
              onChange={(e) => onChange('C', Math.round(10 ** Number(e.target.value) * 1000) / 1000)}
            />
          </label>
          <label className="field">
            <span>Decision threshold: {config.threshold}</span>
            <input
              type="range" min="0" max="1" step="0.01"
              value={config.threshold}
              onChange={(e) => onChange('threshold', Number(e.target.value))}
            />
          </label>
        </div>
      </section>

      <section className="card">
        <h3 className="card-title">Features ({config.features.length}/{features.length} selected)</h3>
        <div className="toolbar">
          <button type="button" onClick={() => onChange('features', features.map((f) => f.name))}>Select all</button>
          <button type="button" onClick={() => onChange('features', [])}>Clear</button>
        </div>
        <div className="features">
          {features.map((feature) => (
            <label key={feature.name} className="checkbox">
              <input
                type="checkbox"
                checked={config.features.includes(feature.name)}
                onChange={() => toggleFeature(feature.name)}
              />
              <span>{feature.name} <small>({feature.missing_pct}%)</small></span>
            </label>
          ))}
        </div>
      </section>
    </>
  )
}