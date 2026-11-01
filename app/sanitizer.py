"""
Request/response processing, kept separate from the HTTP layer so it can be
tested on its own.

Every text field the model will see is scanned - not just the user message:
system prompts, assistant history, tool-call arguments, tool results, tool
definitions and the end-user id. Tool arguments are a common leak path that
naive filters miss.
"""
import copy
import hashlib
import re
from dataclasses import dataclass, field

from .detector import Detector, Finding
from .policy import Policy
from .tokenizer import Vault, rehydrate, repair, tokenize


# JSON Schema and tool fields that are identifiers, not text. Changing them
# would break the tool the model is asked to call.
SCHEMA_KEYS = {"type", "name", "required", "$ref", "format", "pattern"}

# Words in a name that identify nobody on their own.
TITLES = {"mrs", "shri", "smt", "sri", "prof"}

# shortcut: never evicted and assumes one policy per process, fine for the
# handful of tool sets an app uses; add an LRU if clients generate them.
_clean_tools: set[str] = set()


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
    elif content is not None:
        raise UnsupportedContent("message content must be a string or a list of parts")
    # On "tool" messages, name is the function's name, not a participant's.
    # OpenAI rejects "<" and spaces in name, so a masked one loses them.
    if isinstance(message.get("name"), str) and message.get("role") != "tool":
        yield (lambda m=message: m["name"],
               lambda v, m=message: m.__setitem__("name", re.sub(r"[^\w-]+", "", v)))
    calls = [call.get("function", {}) for call in message.get("tool_calls") or []]
    for fn in calls + [message.get("function_call") or {}]:  # function_call: legacy API
        if isinstance(fn.get("arguments"), str):
            yield (lambda f=fn: f["arguments"], lambda v, f=fn: f.__setitem__("arguments", v))
        elif fn.get("arguments") is not None:
            raise UnsupportedContent("tool arguments must be a string")


def _schema_slots(node):
    """Yield (getter, setter) pairs for the free text in tool definitions,
    leaving the parts that make the schema valid untouched."""
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, value in items:
        if isinstance(value, str):
            if key not in SCHEMA_KEYS:
                yield (lambda n=node, k=key: n[k], lambda v, n=node, k=key: n.__setitem__(k, v))
        # A dict here may be a property that happens to be called "name".
        elif isinstance(value, dict) or (isinstance(value, list) and key not in SCHEMA_KEYS):
            yield from _schema_slots(value)


def _request_slots(payload: dict):
    """Yield (getter, setter) pairs for every text field sent to the model."""
    for msg in payload.get("messages", []):
        yield from _text_slots(msg)
    user = payload.get("user")
    if isinstance(user, str):
        yield (lambda p=payload: p["user"], lambda v, p=payload: p.__setitem__("user", v))
    elif user is not None:
        raise UnsupportedContent("user must be a string")


def _tool_slots(payload: dict):
    for key in ("tools", "functions"):  # functions: legacy API
        if payload.get(key) is not None:
            yield from _schema_slots(payload[key])


def _name_parts(known: list[tuple[str, str]]) -> dict[str, str]:
    """Each word of a detected full name -> the value to mask it as: the full
    name, so a lone "Priya" shares <PERSON_1>, unless two people share the word."""
    owners: dict[str, set[str]] = {}
    for etype, value in known:
        words = [w.strip(".,") for w in value.split()]
        if etype == "PERSON" and len(words) > 1:
            for word in words:
                if len(word) > 2 and word.lower() not in TITLES:
                    owners.setdefault(word, set()).add(value)
    return {w: next(iter(names)) if len(names) == 1 else w for w, names in owners.items()}


def _name_word_pattern(word: str) -> re.Pattern:
    # Case-sensitive, unlike full values: "Will Smith" must not mask "you will".
    return re.compile(rf"(?<!\w)(?:{re.escape(word)}|{re.escape(word.upper())})(?!\w)")


def sanitize_request(payload: dict, detector: Detector, policy: Policy) -> SanitizeResult:
    clean = copy.deepcopy(payload)  # never mutate the caller's object
    vault = Vault()
    to_block: set[str] = set()

    # Pass 1: detect everything first, so we can block before tokenizing anything.
    work: list[tuple[callable, list[Finding]]] = []
    def scan(text: str) -> list[Finding]:
        findings = detector.detect(text)
        to_block.update(f.entity_type for f in findings
                        if policy.action_for(f.entity_type) == "block")
        return findings

    for get, set_ in _request_slots(clean):
        work.append((get, set_, scan(get())))

    # Tool definitions can hold hundreds of strings and repeat on every request,
    # so they are scanned as one text; propagation below then masks what was
    # found in each string. Clean ones are remembered by hash, which stores no PII.
    tool_slots = list(_tool_slots(clean))
    tool_findings: list[Finding] = []
    if tool_slots:
        joined = "\n\n".join(get() for get, _ in tool_slots)
        digest = hashlib.sha256(joined.encode()).hexdigest()
        if digest not in _clean_tools:
            tool_findings = scan(joined)
            if not tool_findings:
                _clean_tools.add(digest)
        work += [(get, set_, []) for get, set_ in tool_slots]

    if to_block:
        raise BlockedRequest(to_block)

    # Known-value propagation: NER is not perfectly consistent - it may spot
    # "Priya Nair" in one sentence and miss it in the next. Anything detected
    # ONCE is masked EVERYWHERE it appears in the request. Longest first, so
    # "Priya" can't claim the start of "Priya Nair" and leave "Nair" exposed.
    all_findings = [f for _, _, fs in work for f in fs] + tool_findings
    known = {(f.entity_type, part) for f in all_findings
             for part in f.text.split("\n") if part.strip()}
    known = sorted(known, key=lambda k: len(k[1]), reverse=True)
    # Whole words only: "Raj" must not eat the start of "Rajasthan".
    patterns = [(etype, value, re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE))
                for etype, value in known]
    # After the full values, so "Priya Nair" is masked whole before "Nair" is tried.
    patterns += [("PERSON", value, _name_word_pattern(word))
                 for word, value in _name_parts(known).items()]
    for i, (get, set_, findings) in enumerate(work):
        text = get()
        extra = []
        for etype, value, pattern in patterns:
            for m in pattern.finditer(text):
                start, end = m.span()
                if not any(start < f.end and f.start < end for f in findings + extra):
                    extra.append(Finding(etype, start, end, 1.0, value))
        work[i] = (get, set_, findings + extra)

    # Pass 2: replace PII with placeholders. One vault for the whole request,
    # so "Rahul" in message 1 and message 5 both become <PERSON_1>.
    for get, set_, findings in work:
        if findings:
            set_(tokenize(get(), findings, vault))

    return SanitizeResult(payload=clean, vault=vault,
                          entity_types=[f.entity_type for _, _, fs in work for f in fs])


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
            text = repair(text, vault)
            new_pii = detector.detect(text, ignore=vault)
            found += [f.entity_type for f in new_pii]
            if redact_new_pii:
                for f in sorted(new_pii, key=lambda f: f.start, reverse=True):
                    text = text[: f.start] + f"[REDACTED_{f.entity_type}]" + text[f.end :]
            set_(rehydrate(text, vault))
    return out, found
