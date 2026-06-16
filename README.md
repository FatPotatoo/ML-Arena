# ML Arena

An interactive ML experimentation platform. Users configure a machine-learning
pipeline through a UI (no code), train a model, and read the evaluation metrics —
to learn practical ML concepts (feature selection, preprocessing, imbalance,
regularization, leakage, bias-variance, thresholds) by hands-on experimentation.

This repo is the **V1** build: **Weather dataset + Logistic Regression only.**

---

## V1 scope (frozen)

**In:**
- One dataset: **Weather AUS**, target **`RainTomorrow`** (binary: did it rain the next day).
- One model: **Logistic Regression**.
- **Stateless** — nothing is saved to a database. Configure → train → see metrics.
- A single flat config screen (no beginner/advanced modes).
- User-configurable pipeline **knobs** (see below).
- **Full metric panel** as output (not a single headline score).

**Out (deferred to later phases):**
- Leaderboard, hidden-test ranking, experiment persistence / comparison.
- Insights engine (automated observations).
- Feature engineering (derived-feature expression builder).
- Other models (Decision Tree, Random Forest, KNN, K-Means).
- Auth / user roles, multiple datasets, AutoML.
- Redis / task queue (training is ~1-2s, so requests are synchronous).

---

## The fixed 3-way split

Not a user choice — hardcoded, **chronological** (so the future never trains the past):

| Split | Years | Visible to user? | Role |
|-------|-------|------------------|------|
| Train | 2008-2014 | yes | model is fit here |
| Validation | 2015 | yes (metrics shown) | user tunes against this |
| Test | 2016+ | **HIDDEN** | honest final score; revealed once on "Final evaluation" |

Class balance is ~22% positive and stable across all three splits.

---

## User-configurable knobs (the config schema)

| Stage | Control | Options | Default |
|-------|---------|---------|---------|
| Feature selection | which columns to include | checkboxes over 21 features | all |
| Missing values | strategy | drop row / drop column / impute | impute |
| | impute statistic | mean / median / mode / constant | median |
| | missing-indicator flag | on / off per column | off |
| Encoding | `Location` | one-hot / ordinal / drop | one-hot |
| | wind direction | one-hot / cyclical (sin/cos) / drop | cyclical |
| Scaling | continuous features | none / standard / minmax / robust | standard |
| Imbalance | class weighting | none / balanced | balanced |
| **Leakage policy** | when transforms are fit | **prevent** (train only) / **allow** (train+val) | prevent |
| Logistic Regression | `C` (regularization strength) | slider (log scale) | 1.0 |
| | penalty | L1 / L2 | L2 |
| | decision threshold | slider 0-1 | 0.5 |

**Locked sub-decisions:** global-only imputation (no group-based fallback);
`class_weight` only (no SMOTE/resampling); L1 + L2 penalties (no elasticnet);
threshold is a user slider defaulting to 0.5.

**Leakage knob detail:** "prevent" fits imputer/scaler/encoder on **train only**;
"allow" fits them on **train + validation** (the classic mistake of preprocessing
before splitting). It never touches the hidden test set. The hidden test is the
yardstick that reveals an inflated validation score when leakage is on.

---

## Tech stack

**Backend (built first):**
- Python 3.13, **FastAPI** (web framework), **Pydantic** (config validation), **Uvicorn** (server).
- **scikit-learn**, **pandas**, **numpy** (the ML pipeline), **joblib** (caching).

**Frontend (not started):**
- **React** (via Vite), **Tailwind CSS**, **shadcn/ui** or Headless UI (form controls),
  **TanStack Query** (data fetching), **Axios**, **Recharts** (metric charts).

**Infra:** Docker + docker-compose; Render/Railway for hosting (deferred). No database in V1.

---

## Running the backend

From `backend/`:

```
python -m venv .venv                      # one-time: create the isolated env
.venv\Scripts\python.exe -m pip install -r requirements.txt   # one-time: install deps
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Then open **http://localhost:8000/docs** — FastAPI's auto-generated interactive API page.

Health check: http://localhost:8000/api/health → `{"status": "ok"}`

---

## Current status

**Increment 1 — DONE.** Project skeleton + running server with two read endpoints:
- `GET /api/health` — heartbeat.
- `GET /api/dataset/info` — dataset summary (rows, positive rate, split sizes,
  and every feature with its missing-%). The frontend uses this to build the
  feature-selection checkboxes.

**Increment 2 — NEXT.** `POST /api/train`:
1. Define the Pydantic **config schema** (all knobs above).
2. Build the scikit-learn **pipeline from the config** (ColumnTransformer +
   LogisticRegression, branching on the leakage policy).
3. Train on the chosen splits, evaluate, return the **full metric panel**
   (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix) for
   train + validation. Test stays hidden until a "Final evaluation" call.

---

## Reference

The pipeline logic is adapted from a prototype notebook (`Rainfall.ipynb` in the
sibling `RainfallAnalysis` project), which worked through the full preprocessing,
a Logistic Regression baseline, and a polynomial-feature experiment that confirmed
the model is **bias-limited (underfit), not overfit** — useful intuition for the
defaults chosen above.
