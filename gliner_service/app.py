"""
NVIDIA GLiNER-PII as a small HTTP service, for Google Cloud Run.

The gateway's demo page calls this for its comparison panel only. It keeps
nothing: no logs of the text, no storage.
"""
from fastapi import FastAPI
from gliner import GLiNER
from pydantic import BaseModel

model = GLiNER.from_pretrained("nvidia/gliner-pii")
app = FastAPI(title="GLiNER-PII")


class DetectRequest(BaseModel):
    text: str
    labels: list[str]
    threshold: float = 0.5


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect(request: DetectRequest):
    return model.predict_entities(request.text, request.labels, threshold=request.threshold)
