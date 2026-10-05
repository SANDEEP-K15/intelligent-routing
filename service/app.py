"""Local routing service.

Run:  uvicorn service.app:app --port 8000
  GET  /             one-page UI
  GET  /api/health   model status
  POST /api/route    route one service request

No external API or key is used. If the model artefact is missing the service still starts and
answers 503 with instructions instead of crashing.
"""

from __future__ import annotations

import os
import time
from enum import Enum, StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel_router import __version__, config
from kestrel_router.model import ModelArtefactError, RouterModel, load_model

STATIC_DIR = Path(__file__).parent / "static"
MAX_TEXT_LENGTH = 2000


class Channel(StrEnum):
    ivr = "ivr"
    chat = "chat"
    whatsapp = "whatsapp"
    email = "email"


class Warranty(StrEnum):
    in_warranty = "in_warranty"
    shield = "shield"
    out_of_warranty = "out_of_warranty"


ProductFamily = Enum(  # type: ignore[misc]
    "ProductFamily", {p.replace(" ", "_").lower(): p for p in config.PRODUCT_FAMILIES}, type=str
)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH,
                              description="Customer's opening message or IVR transcript")
    channel: Optional[Channel] = None
    warranty_status: Optional[Warranty] = None
    product_family: Optional[ProductFamily] = None  # type: ignore[valid-type]

    @field_validator("request_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("request_text must contain some text")
        return v


class Alternative(BaseModel):
    team: str
    probability: float


class RouteResponse(BaseModel):
    team: str
    confidence: float
    confidence_band: str
    reasons: list[str]
    alternatives: list[Alternative]
    inputs_used: list[str]
    model_version: str
    latency_ms: float


def _model_path() -> Path:
    return Path(os.environ.get("KESTREL_MODEL_PATH", config.MODEL_PATH))


@lru_cache(maxsize=1)
def get_model() -> RouterModel:
    return load_model(_model_path())


app = FastAPI(
    title="Kestrel Home request routing",
    version=__version__,
    description="Suggests the team queue for a new service request. Runs locally; no paid API.",
)


@app.get("/api/health")
def health() -> dict:
    try:
        model = get_model()
    except ModelArtefactError as exc:
        return {"status": "model_missing", "detail": str(exc)}
    return {
        "status": "ok",
        "model": model.metadata.get("spec", {}).get("name"),
        "trained_rows": model.metadata.get("training_rows"),
        "inputs_used": model.metadata.get("input_fields"),
        "teams": model.classes,
        "validation": model.metadata.get("validation"),
        "cost_per_prediction_rs": config.MODEL_API_COST_PER_PREDICTION,
    }


@app.post("/api/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    try:
        model = get_model()
    except ModelArtefactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    t0 = time.perf_counter()
    result = model.route(
        req.request_text,
        channel=req.channel.value if req.channel else None,
        warranty_status=req.warranty_status.value if req.warranty_status else None,
        product_family=req.product_family.value if req.product_family else None,
    )
    inputs_used = model.metadata.get("input_fields", ["request_text"])
    missing = [f for f in inputs_used if f != "request_text" and getattr(req, f) is None]
    if missing:
        result["reasons"].append(
            f"The {' and '.join(m.replace('_', ' ') for m in missing)} was not provided, so the suggestion "
            "relies on the message text alone. Adding it can improve accuracy."
        )
    return RouteResponse(
        **result,
        inputs_used=inputs_used,
        model_version=str(model.metadata.get("trained_at_utc", "unknown")),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
