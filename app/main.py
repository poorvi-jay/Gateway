"""
LLM gateway - phase 1: a plain pass-through proxy.

Exposes an OpenAI-compatible POST /v1/chat/completions and forwards requests
unchanged to the upstream LLM. PII handling is added in later phases.
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "mock")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL")

app = FastAPI(title="PII-safe LLM Gateway")


async def call_upstream(payload: dict) -> dict:
    """Forward the payload to the real LLM provider."""
    if UPSTREAM_BASE_URL == "mock":
        # Offline mode for demos/tests: echoes what the "LLM" received.
        last = next((m.get("content") for m in reversed(payload["messages"])
                     if m.get("role") == "user"), "")
        return {
            "id": "mock-1", "object": "chat.completion", "model": "mock",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": f"[mock LLM] I received: {last}"}}],
        }
    if UPSTREAM_MODEL:
        payload = {**payload, "model": UPSTREAM_MODEL}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{UPSTREAM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {UPSTREAM_API_KEY}"},
        )
        r.raise_for_status()
        return r.json()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    try:
        return await call_upstream(payload)
    except Exception:
        return JSONResponse(status_code=502, content={"error": {
            "message": "Upstream LLM request failed", "type": "upstream_error",
            "code": "upstream_error"}})
