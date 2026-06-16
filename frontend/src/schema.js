// The frontend mirror of the backend's TrainConfig. Defaults match the schema's
// defaults so an untouched form trains the same baseline as POST /api/train {}.
export const DEFAULT_CONFIG = {
  // features starts empty; App fills it with "all features" once the dataset
  // info arrives (so every box is checked by default, like the backend default).
  features: [],
  // PER-FEATURE missing handling: { featureName: { strategy, statistic } }.
  // App fills this once dataset info arrives (one entry per feature that has gaps).
  missing: {},
  location_encoding: 'onehot',
  wind_encoding: 'cyclical',
  scaling: 'standard',
  class_weight: 'balanced',
  leakage_policy: 'prevent',
  C: 1.0,
  penalty: 'L2',
  threshold: 0.5,
}

// Per-feature missing options (used by the missing-values section, not OPTIONS).
export const MISSING_STRATEGY = [['impute', 'Impute'], ['drop_row', 'Drop rows'], ['drop_col', 'Exclude feature']]
export const STATISTIC_NUMERIC = [['median', 'Median'], ['mean', 'Mean'], ['mode', 'Most frequent'], ['constant', 'Constant (0)']]
export const STATISTIC_TEXT = [['mode', 'Most frequent'], ['constant', 'Constant']]

// The default impute statistic for a feature, by its type.
export const defaultStatistic = (type) => (type === 'numeric' ? 'median' : 'mode')

// For each dropdown knob: a list of [value-sent-to-backend, label-shown-to-user].
// The values exactly match the Literal options in schemas.py.
export const OPTIONS = {
  location_encoding: [['onehot', 'One-hot'], ['ordinal', 'Ordinal'], ['drop', 'Drop']],
  wind_encoding: [['cyclical', 'Cyclical (sin/cos)'], ['onehot', 'One-hot'], ['drop', 'Drop']],
  scaling: [['standard', 'Standard'], ['minmax', 'Min-max'], ['robust', 'Robust'], ['none', 'None']],
  class_weight: [['balanced', 'Balanced'], ['none', 'None']],
  leakage_policy: [['prevent', 'Prevent (fit on train only)'], ['allow', 'Allow (fit on train + val)']],
  penalty: [['L2', 'L2 (ridge)'], ['L1', 'L1 (lasso)']],
}
