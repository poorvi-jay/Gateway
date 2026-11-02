"""
PII detection: two layers combined in one Presidio AnalyzerEngine.

  Layer 1 - regex + checksum recognizers (emails, cards via Luhn, Aadhaar via
            Verhoeff, PAN, phones, UPI, IPs, other Indian IDs, secrets). Fast
            and deterministic.
  Layer 2 - spaCy NER model for things with no fixed shape (names, places).

Presidio runs both and returns every match with a confidence score; this module
filters by score and resolves overlapping matches.
"""
import os
import re
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from .policy import Policy
from .recognizers import custom_recognizers

PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z_]*_\d+>")

# NER sometimes pulls the verb into the name: "[Email Rahul Sharma] at ...".
PERSON_LEAD_RE = re.compile(
    r"(?:e-?mail|mail|call|phone|text|message|contact|ping|sms|whatsapp)\s+", re.IGNORECASE)

# Business terms NER mistakes for places.
ALLOWED_TERMS = ["Q1", "Q2", "Q3", "Q4"]


@dataclass
class Finding:
    entity_type: str
    start: int
    end: int
    score: float
    text: str


class Detector:
    def __init__(self, policy: Policy, spacy_model: str | None = None):
        self.policy = policy
        model = spacy_model or os.getenv("SPACY_MODEL", "en_core_web_lg")
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model}],
            }
        ).create_engine()
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"],
            # Whole words, or "pin" matches "shopping". Suffix for "A1234567 is my passport".
            context_aware_enhancer=LemmaContextAwareEnhancer(
                context_suffix_count=3, context_matching_mode="whole_word"),
        )
        self.analyzer.registry.remove_recognizer("CreditCardRecognizer")  # see CardRecognizer
        for rec in custom_recognizers():
            self.analyzer.registry.add_recognizer(rec)

    def _analyze(self, text: str):
        return self.analyzer.analyze(
            text=text,
            language="en",
            entities=self.policy.scanned_entities,
            score_threshold=self.policy.score_threshold,
            allow_list=ALLOWED_TERMS,
        )

    def detect(self, text: str, ignore=()) -> list[Finding]:
        """`ignore` holds placeholders we issued ourselves. Only those may be
        skipped: anything shaped like a placeholder could be hiding real PII."""
        if not text or not text.strip():
            return []
        results = self._analyze(text)
        # Regexes treat "_" as part of a word, so an ID glued to one
        # (CARD_4111111111111111) slips past them. Offsets stay the same.
        if "_" in text:
            results += self._analyze(text.replace("_", " "))
        placeholder_spans = [m.span() for m in PLACEHOLDER_RE.finditer(text)
                             if m.group(0) in ignore]
        findings = [
            _finding(r, text)
            for r in results
            if not _overlaps_any(r.start, r.end, placeholder_spans)
        ]
        return self._resolve_overlaps(findings)

    def _resolve_overlaps(self, findings: list[Finding]) -> list[Finding]:
        """When two matches overlap (e.g. PHONE_NUMBER and IN_PHONE on the same
        digits), keep one. Priority: 'block' and 'redact' entities first (never
        let a weaker label hide something stricter), then higher score, then
        longer span."""
        def priority(f: Finding):
            strict = self.policy.action_for(f.entity_type) in ("block", "redact")
            return (strict, f.score, f.end - f.start)

        kept: list[Finding] = []
        for f in sorted(findings, key=priority, reverse=True):
            if not any(f.start < k.end and k.start < f.end for k in kept):
                kept.append(f)
        return sorted(kept, key=lambda f: f.start)


def _finding(result, text: str) -> Finding:
    start = result.start
    if result.entity_type == "PERSON":
        lead = PERSON_LEAD_RE.match(text, start, result.end)
        if lead and lead.end() < result.end:
            start = lead.end()
    return Finding(result.entity_type, start, result.end, result.score, text[start:result.end])


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in spans)
