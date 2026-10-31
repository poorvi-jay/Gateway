"""
PII-safe LLM gateway.

Exposes an OpenAI-compatible POST /v1/chat/completions. Apps point their
existing OpenAI client at this server (change base_url only); the gateway
strips PII, forwards to the real LLM, and restores values in the answer.

    App -> [ auth -> detect -> policy -> tokenize ] -> LLM
    App <- [ rehydrate <- scan response           ] <- LLM
"""
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .audit import audit
from .detector import Detector
from .policy import load_policy
from .sanitizer import (BlockedRequest, UnsupportedContent, process_response,
                        sanitize_request)

POLICY_PATH = os.getenv("POLICY_PATH", "policy.yaml")
UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "mock")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL")          # optional model override
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")        # optional client auth
DETECTION_TIMEOUT_S = float(os.getenv("DETECTION_TIMEOUT_S", "5"))
REDACT_RESPONSE_PII = os.getenv("REDACT_RESPONSE_PII", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the NER model once at startup, not per request.
    app.state.policy = load_policy(POLICY_PATH)
    app.state.detector = Detector(app.state.policy)
    yield


app = FastAPI(title="PII-safe LLM Gateway", lifespan=lifespan)


def error(status: int, message: str, code: str) -> JSONResponse:
    # Same error shape as OpenAI so existing clients surface it correctly.
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": code, "code": code}})


async def call_upstream(payload: dict) -> dict:
    """Forward the SANITIZED payload to the real LLM provider."""
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


def check_auth(request: Request) -> bool:
    if not GATEWAY_API_KEY:
        return True
    return request.headers.get("authorization") == f"Bearer {GATEWAY_API_KEY}"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    rid = str(uuid.uuid4())
    t0 = time.perf_counter()

    if not check_auth(request):
        return error(401, "Invalid gateway API key", "unauthorized")

    try:
        payload = await request.json()
    except Exception:
        return error(400, "Body must be JSON", "invalid_request")

    if payload.get("stream"):
        return error(400, "Streaming is not supported by this gateway yet", "unsupported")

    detector, policy = request.app.state.detector, request.app.state.policy

    # ---- Request path. FAIL CLOSED: any failure here blocks the request. ----
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(sanitize_request, payload, detector, policy),
            timeout=DETECTION_TIMEOUT_S,
        )
    except BlockedRequest as e:
        audit(rid, "blocked", entities_in=e.entity_types, reason="policy_block")
        return error(400, f"Request blocked: contains {', '.join(e.entity_types)}",
                     "pii_blocked")
    except UnsupportedContent as e:
        audit(rid, "blocked", reason="uninspectable_content")
        return error(400, str(e), "unsupported_content")
    except Exception as e:
        # Includes timeouts. Never forward a request we failed to inspect.
        audit(rid, "error", reason=type(e).__name__)
        return error(503, "PII inspection failed; request was not sent", "inspection_failed")

    gateway_ms = (time.perf_counter() - t0) * 1000

    # ---- Forward sanitized payload ----
    try:
        upstream = await call_upstream(result.payload)
    except Exception as e:
        audit(rid, "error", entities_in=result.entity_types, reason=f"upstream:{type(e).__name__}")
        return error(502, "Upstream LLM request failed", "upstream_error")

    # ---- Response path ----
    t1 = time.perf_counter()
    try:
        final, resp_entities = await asyncio.to_thread(
            process_response, upstream, result.vault, detector, REDACT_RESPONSE_PII)
    except Exception as e:
        audit(rid, "error", entities_in=result.entity_types, reason=f"response:{type(e).__name__}")
        return error(503, "Response inspection failed", "inspection_failed")
    gateway_ms += (time.perf_counter() - t1) * 1000

    audit(rid, "forwarded", entities_in=result.entity_types,
          entities_out=resp_entities, latency_ms=gateway_ms)
    return JSONResponse(final, headers={"x-gateway-request-id": rid,
                                        "x-gateway-latency-ms": f"{gateway_ms:.1f}"})


@app.post("/debug/sanitize")
async def debug_sanitize(request: Request):
    """Shows exactly what the LLM would receive. For demos only - disable in
    production (set ENABLE_DEBUG=false)."""
    if os.getenv("ENABLE_DEBUG", "true").lower() != "true":
        return error(404, "Not found", "not_found")
    if not check_auth(request):
        return error(401, "Invalid gateway API key", "unauthorized")
    payload = await request.json()
    try:
        result = sanitize_request(payload, request.app.state.detector, request.app.state.policy)
    except BlockedRequest as e:
        return {"blocked": True, "entity_types": e.entity_types}
    return {"blocked": False, "sent_to_llm": result.payload["messages"],
            "entity_types": result.entity_types}
