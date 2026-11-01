"""
Optional comparison: what NVIDIA GLiNER-PII would mask in the same text.

Runs on this machine, never through NVIDIA's hosted API, which would send the
raw PII out. It is shown beside the real result only: nothing here changes what
reaches the LLM. Needs requirements-gliner.txt and ENABLE_GLINER=true.
"""
from .detector import Finding
from .tokenizer import Vault, tokenize

MODEL = "nvidia/gliner-pii"
THRESHOLD = 0.5  # same as score_threshold in policy.yaml

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
    "tax_id": "IN_PAN",
    "api_key": "API_KEY",
    "password": "PASSWORD",
    "ipv4": "IP_ADDRESS",
}


class GlinerDetector:
    def __init__(self):
        from gliner import GLiNER  # imported here so the normal install doesn't need it
        self.model = GLiNER.from_pretrained(MODEL)

    def detect(self, text: str) -> list[Finding]:
        found = self.model.predict_entities(text, list(LABELS), threshold=THRESHOLD)
        return [Finding(LABELS[f["label"]], f["start"], f["end"], f["score"], text[f["start"]:f["end"]])
                for f in found]


def mask(text: str, detector) -> tuple[str, list[str]]:
    findings = detector.detect(text)
    return tokenize(text, findings, Vault()), [f.entity_type for f in findings]
