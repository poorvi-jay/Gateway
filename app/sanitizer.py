"""
Request/response processing, kept separate from the HTTP layer so it can be
tested on its own.

Every text field the model will see is scanned - not just the user message:
system prompts, assistant history, tool-call arguments and tool results. Tool
arguments are a common leak path that naive filters miss.
"""
import copy
from dataclasses import dataclass, field

from .detector import Detector, Finding
from .policy import Policy
from .tokenizer import Vault, rehydrate, tokenize


class BlockedRequest(Exception):
    def __init__(self, entity_types: set[str]):
        self.entity_types = sorted(entity_types)
        super().__init__(f"blocked: {self.entity_types}")


class UnsupportedContent(Exception):
    """Content we cannot inspect (e.g. images). Rejected rather than passed
    through unchecked - the gateway fails closed."""


@dataclass
class SanitizeResult:
    payload: dict
    vault: Vault
    entity_types: list[str] = field(default_factory=list)


def _text_slots(message: dict):
    """Yield (getter, setter) pairs for every text field in a chat message."""
    content = message.get("content")
    if isinstance(content, str):
        yield (lambda m=message: m["content"],
               lambda v, m=message: m.__setitem__("content", v))
    elif isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                yield (lambda p=part: p["text"], lambda v, p=part: p.__setitem__("text", v))
            else:
                raise UnsupportedContent(
                    f"content part of type '{part.get('type')}' cannot be inspected for PII"
                )
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        if isinstance(fn.get("arguments"), str):
            yield (lambda f=fn: f["arguments"], lambda v, f=fn: f.__setitem__("arguments", v))


def sanitize_request(payload: dict, detector: Detector, policy: Policy) -> SanitizeResult:
    clean = copy.deepcopy(payload)  # never mutate the caller's object
    vault = Vault()
    seen: list[str] = []
    to_block: set[str] = set()

    # Pass 1: detect everything first, so we can block before tokenizing anything.
    work: list[tuple[callable, list[Finding]]] = []
    for msg in clean.get("messages", []):
        for get, set_ in _text_slots(msg):
            findings = detector.detect(get())
            for f in findings:
                seen.append(f.entity_type)
                if policy.action_for(f.entity_type) == "block":
                    to_block.add(f.entity_type)
            work.append((get, set_, findings))

    if to_block:
        raise BlockedRequest(to_block)

    # Known-value propagation: NER is not perfectly consistent - it may spot
    # "Priya Nair" in one sentence and miss it in the next. Anything detected
    # ONCE is masked EVERYWHERE it appears in the request.
    known = {(f.entity_type, f.text) for _, _, fs in work for f in fs}
    for i, (get, set_, findings) in enumerate(work):
        text = get()
        extra = []
        for etype, value in known:
            start = text.find(value)
            while start != -1:
                end = start + len(value)
                if not any(start < f.end and f.start < end for f in findings + extra):
                    extra.append(Finding(etype, start, end, 1.0, value))
                    seen.append(etype)
                start = text.find(value, end)
        work[i] = (get, set_, findings + extra)

    # Pass 2: replace PII with placeholders. One vault for the whole request,
    # so "Rahul" in message 1 and message 5 both become <PERSON_1>.
    for get, set_, findings in work:
        if findings:
            set_(tokenize(get(), findings, vault))

    return SanitizeResult(payload=clean, vault=vault, entity_types=seen)


def process_response(response: dict, vault: Vault, detector: Detector,
                     redact_new_pii: bool) -> tuple[dict, list[str]]:
    """Scan the model's answer for PII it produced itself (not from our
    placeholders), optionally redact it, then restore our placeholders."""
    out = copy.deepcopy(response)
    found: list[str] = []
    for choice in out.get("choices", []):
        msg = choice.get("message") or {}
        for get, set_ in _text_slots(msg):
            text = get()
            if not text:
                continue
            new_pii = detector.detect(text)  # placeholders are skipped by the detector
            found += [f.entity_type for f in new_pii]
            if redact_new_pii:
                for f in sorted(new_pii, key=lambda f: f.start, reverse=True):
                    text = text[: f.start] + f"[REDACTED_{f.entity_type}]" + text[f.end :]
            set_(rehydrate(text, vault))
    return out, found
