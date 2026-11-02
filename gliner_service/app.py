"""
NVIDIA GLiNER-PII as a small HTTP service, for Google Cloud Run.

The gateway's demo page calls this for its comparison panel only. It keeps
nothing: no logs of the text, no storage. GLINER_TOKEN is shared with the
gateway, so nobody else can spend the free allowance.
"""
import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from gliner import GLiNER
from pydantic import BaseModel, Field

TOKEN = os.environ["GLINER_TOKEN"]  # no default: an open service must not start by accident
model = GLiNER.from_pretrained("nvidia/gliner-pii")
app = FastAPI(title="GLiNER-PII")


class DetectRequest(BaseModel):
    text: str = Field(max_length=200_000)
    labels: list[str] = Field(max_length=50)
    threshold: float = 0.5


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect(request: DetectRequest, authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {TOKEN}".encode()):
        raise HTTPException(401, "Invalid token")
    return model.predict_entities(request.text, request.labels, threshold=request.threshold)
