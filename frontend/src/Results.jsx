// Displays the metric panel returned by POST /api/train. Pure presentation —
// it receives `result` as a prop and just renders it.

// One labelled number.
function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="metric-value">{value.toFixed(3)}</span>
      <span className="metric-label">{label}</span>
    </div>
  )
}

// The 2x2 confusion matrix as a small table.
function Confusion({ cm }) {
  return (
    <table className="confusion">
      <thead>
        <tr><th></th><th>Pred No</th><th>Pred Yes</th></tr>
      </thead>
      <tbody>
        <tr><th>Actual No</th><td>{cm.tn.toLocaleString()}</td><td>{cm.fp.toLocaleString()}</td></tr>
        <tr><th>Actual Yes</th><td>{cm.fn.toLocaleString()}</td><td>{cm.tp.toLocaleString()}</td></tr>
      </tbody>
    </table>
  )
}

// One split's scorecard (train or validation).
function SplitMetrics({ title, m }) {
  return (
    <div className="split-metrics">
      <h4>{title} <small>({m.n.toLocaleString()} rows · {Math.round(m.positive_rate * 100)}% rain)</small></h4>
      <div className="metric-grid">
        <Metric label="Accuracy" value={m.accuracy} />
        <Metric label="Precision" value={m.precision} />
        <Metric label="Recall" value={m.recall} />
        <Metric label="F1" value={m.f1} />
        <Metric label="ROC-AUC" value={m.roc_auc} />
        <Metric label="PR-AUC" value={m.pr_auc} />
      </div>
      <Confusion cm={m.confusion_matrix} />
    </div>
  )
}

export default function Results({ result }) {
  return (
    <section className="card">
      <h3 className="card-title">Results <small>· {result.n_features_in} features after encoding</small></h3>
      <SplitMetrics title="Validation" m={result.validation} />
      <SplitMetrics title="Train" m={result.train} />
      <p className="hint">
        Validation is the honest score to tune against. A big Train-vs-Validation
        gap suggests overfitting; the hidden test set is revealed only at final evaluation.
      </p>
    </section>
  )
}
