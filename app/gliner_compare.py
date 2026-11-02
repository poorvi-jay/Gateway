"""
Optional comparison: what NVIDIA GLiNER-PII would mask in the same text.

Runs on this machine (requirements-gliner.txt and ENABLE_GLINER=true) or in our
own Cloud Run service (GLINER_URL, see gliner_service/), never through NVIDIA's
hosted API. It is shown beside the real result only: nothing here changes what
reaches the LLM.
"""
import os
import re

import httpx

from .detector import Finding
from .recognizers import verhoeff_valid
from .tokenizer import Vault, tokenize

MODEL = "nvidia/gliner-pii"
THRESHOLD = 0.5  # same as score_threshold in policy.yaml
REMOTE_TIMEOUT_S = 20  # the demo's Output panel waits for this one

# Label the model is asked for -> placeholder type, matching ours where one exists.
LABELS = {
    "person": "PERSON",
    "email": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "street_address": "LOCATION",
    "city": "LOCATION",
    "postcode": "PIN_CODE",
    "date_of_birth": "DATE_OF_BIRTH",
    "credit_debit_card": "CREDIT_CARD",
    "account_number": "BANK_ACCOUNT",
    "bank_routing_number": "IN_IFSC",
    "national_id": "NATIONAL_ID",
    "aadhaar_number": "NATIONAL_ID",
    "pan_number": "NATIONAL_ID",
    "tax_id": "NATIONAL_ID",
    "api_key": "API_KEY",
    "password": "PASSWORD",
    "ipv4": "IP_ADDRESS",
}

PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")


def entity_type(label: str, value: str) -> str:
    """The model calls these a card or a password as often as an ID, so the
    value's shape decides."""
    aadhaar = AADHAAR_RE.search(value)
    if PAN_RE.search(value) or (aadhaar and verhoeff_valid(re.sub(r"\D", "", aadhaar.group()))):
        return "NATIONAL_ID"
    return LABELS[label]


class RemoteModel:
    """The model in our Cloud Run service, called like the local one."""
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def predict_entities(self, text: str, labels: list[str], threshold: float) -> list[dict]:
        r = httpx.post(f"{self.url}/detect", timeout=REMOTE_TIMEOUT_S,
                       headers={"Authorization": f"Bearer {os.getenv('GLINER_TOKEN', '')}"},
                       json={"text": text, "labels": labels, "threshold": threshold})
        r.raise_for_status()
        return r.json()


class GlinerDetector:
    def __init__(self, url: str | None = None):
        if url:
            self.model = RemoteModel(url)
            return
        from gliner import GLiNER  # imported here so the normal install doesn't need it
        self.model = GLiNER.from_pretrained(MODEL)

    def detect(self, text: str) -> list[Finding]:
        found = self.model.predict_entities(text, list(LABELS), threshold=THRESHOLD)
        return [Finding(entity_type(f["label"], text[f["start"]:f["end"]]), f["start"], f["end"],
                        f["score"], text[f["start"]:f["end"]]) for f in found]


def mask(text: str, detector) -> tuple[str, list[str]]:
    findings = detector.detect(text)
    return tokenize(text, findings, Vault()), [f.entity_type for f in findings]
