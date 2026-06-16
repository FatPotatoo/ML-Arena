import { OPTIONS } from './schema'

// A small reusable dropdown. Instead of writing <select> markup five times, we
// write it once and reuse it. `field` is the config key it edits (e.g. "scaling").
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

// The whole form. It owns NO state of its own — it receives `config` from the
// parent and calls `onChange(field, value)` to report edits upward. This pattern
// is called "lifting state up": one owner (App) holds the truth, the form just
// displays it and requests changes.
export default function ConfigForm({ config, features, onChange }) {
  // Add or remove one feature name from the config.features array.
  const toggleFeature = (name) => {
    const next = config.features.includes(name)
      ? config.features.filter((f) => f !== name) // remove
      : [...config.features, name]                 // add
    onChange('features', next)
  }

  return (
    <section className="card">
      <h2>Configure</h2>

      <h3>Preprocessing</h3>
      <div className="grid">
        <Select label="Missing values" field="missing_strategy" config={config} onChange={onChange} />
        {/* The impute-statistic dropdown only matters when strategy is "impute". */}
        {config.missing_strategy === 'impute' && (
          <Select label="Impute statistic" field="impute_statistic" config={config} onChange={onChange} />
        )}
        <Select label="Location encoding" field="location_encoding" config={config} onChange={onChange} />
        <Select label="Wind encoding" field="wind_encoding" config={config} onChange={onChange} />
        <Select label="Scaling" field="scaling" config={config} onChange={onChange} />
      </div>

      <h3>Model</h3>
      <div className="grid">
        <Select label="Class weight" field="class_weight" config={config} onChange={onChange} />
        <Select label="Leakage policy" field="leakage_policy" config={config} onChange={onChange} />
        <Select label="Penalty" field="penalty" config={config} onChange={onChange} />
      </div>

      <div className="grid">
        {/* C is a log-scale slider: the handle moves in powers of 10, so the
            stored value is 10^(slider position). Range -3..3 => C from 0.001 to 1000. */}
        <label className="field">
          <span>Regularization C: {config.C}</span>
          <input
            type="range" min="-3" max="3" step="0.1"
            value={Math.log10(config.C)}
            onChange={(e) => onChange('C', Math.round(10 ** Number(e.target.value) * 1000) / 1000)}
          />
        </label>

        {/* threshold is a plain 0..1 slider. */}
        <label className="field">
          <span>Decision threshold: {config.threshold}</span>
          <input
            type="range" min="0" max="1" step="0.01"
            value={config.threshold}
            onChange={(e) => onChange('threshold', Number(e.target.value))}
          />
        </label>
      </div>

      <h3>Features ({config.features.length}/{features.length} selected)</h3>
      <div className="feature-actions">
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
            <span>{feature.name} <small>({feature.missing_pct}% missing)</small></span>
          </label>
        ))}
      </div>
    </section>
  )
}
