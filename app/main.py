"""
The application entry point.

`app` is the FastAPI application object. The server (uvicorn) imports this
`app` and runs it. Everything wires together here:
  - CORS  : permission for the browser frontend to call this API
  - routers: the actual URL groups (dataset, train, ...)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import dataset

app = FastAPI(title="ML Arena API", version="0.1.0")

# --- CORS ---------------------------------------------------------------
# Browsers block a web page from calling a server on a different address
# unless that server explicitly allows it. Our React dev server runs on
# http://localhost:5173, so we whitelist it here. Without this, the
# frontend's requests would be rejected by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ------------------------------------------------------------
app.include_router(dataset.router)


@app.get("/api/health")
def health():
    """A trivial endpoint to confirm the server is alive. Visiting
    http://localhost:8000/api/health should return {"status": "ok"}."""
    return {"status": "ok"}
