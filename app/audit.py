"""
Audit log: one JSON line per request recording WHAT was found and WHAT was
done - entity types and counts only. Raw PII values are never written here;
otherwise the log file would become a new place for PII to leak from.
"""
import json
import os
import threading
import time
from collections import Counter

_lock = threading.Lock()


def audit(request_id: str, outcome: str, entities_in=None, entities_out=None,
          latency_ms: float | None = None, reason: str | None = None):
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": request_id,
        "outcome": outcome,  # forwarded | blocked | error
        "entities_in": dict(Counter(entities_in or [])),
        "entities_in_response": dict(Counter(entities_out or [])),
        "gateway_latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "reason": reason,
    }
    path = os.getenv("AUDIT_LOG_PATH", "audit.log")
    with _lock, open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
