"""
Training endpoint.

Owns everything under /api/train. The heavy lifting lives in app/training.py;
this file is just the thin "waiter" layer: accept the request, hand it to
run_training, and return the result (or a clean error).
"""
from fastapi import APIRouter, HTTPException

from ..schemas import TrainConfig
from ..training import run_training

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("")
def train(config: TrainConfig) -> dict:
    """POST /api/train -> train one model and return its full metric panel.

    Because the parameter is type-annotated as `TrainConfig`, FastAPI automatically
    reads the JSON request body, validates it against the schema (filling defaults,
    rejecting bad values with a 422), and only then calls this function with a
    ready-to-use `config` object. So an empty body `{}` trains the baseline.
    """
    try:
        return run_training(config)
    except ValueError as exc:
        # e.g. drop_col removed every feature. 400 = "your request was invalid".
        raise HTTPException(status_code=400, detail=str(exc)) from exc
