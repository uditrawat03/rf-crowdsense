from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="RF CrowdSense API", version="0.2.0")


class ActivityRequest(BaseModel):
    channel_occupancy: float = Field(ge=0.0, le=1.0)
    noise_floor_db: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/estimate")
def estimate(request: ActivityRequest) -> dict:
    score = max(0.0, min(1.0, request.channel_occupancy))
    return {
        "activity_score": score,
        "note": "Heuristic API placeholder. It does not identify or track devices.",
    }
