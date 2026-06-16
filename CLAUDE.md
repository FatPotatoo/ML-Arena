# CLAUDE.md — context for AI sessions in this project

This file is loaded automatically when Claude Code starts in this folder. It
carries over the context from the design conversation that produced this project.

## What this is
**ML Arena** — an interactive ML experimentation platform (configure a pipeline
via UI → train → read metrics). **V1 = Weather dataset + Logistic Regression only.**
Read `README.md` for the full frozen scope, the knob list, and the tech stack.

## The user
- **Learning backend web development for the first time.** Explain backend
  concepts (servers, requests, endpoints, routers, ports) in plain terms with
  analogies. Do not assume prior web knowledge.
- Comfortable with ML / scikit-learn (built the prototype notebook).
- Prefers **frequent, small increments** with an explanation after each, rather
  than large code dumps.

## How to work here
- **Backend first**, then frontend.
- Keep V1 simple and **stateless** — no database, no auth, no leaderboard.
- Work in small checkpoints; after each, summarize what was built in beginner terms.
- The fixed chronological split and the leakage-policy knob are core design
  decisions — see README. Do not add a split-strategy knob or an insights engine.

## Conventions
- Backend lives in `backend/`. Virtual env at `backend/.venv` (gitignored).
- One topic per router file under `app/routers/`; `app/data.py` is the single
  place that reads the CSV and performs the split.
- Run the server: `backend/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000`.

## Status
- **Increment 1 done:** running FastAPI server; `GET /api/health`,
  `GET /api/dataset/info`. Verified working.
- **Increment 2 next:** `POST /api/train` — Pydantic config schema, build the
  sklearn pipeline from the config (branching on leakage policy), train, return
  the full metric panel (train + validation; test hidden).

## Decision log (from the design discussion)
- Stateless V1, no DB (chosen for simplicity — user is new to backend).
- 3-way chronological split, test hidden, revealed once on "Final evaluation".
- Output is the full metric panel, never a single headline number.
- Imputation: global-only statistics. Imbalance: `class_weight` only (no resampling).
- Penalties: L1 + L2 (no elasticnet). Threshold: user slider, default 0.5.
- Cut from V1: leaderboard, insights engine, feature engineering, depth ladder,
  split-strategy knob, auth.
