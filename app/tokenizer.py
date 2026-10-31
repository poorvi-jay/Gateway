"""
Reversible pseudonymization.

  tokenize:  "Call Rahul at 9876543210"  ->  "Call <PERSON_1> at <IN_PHONE_1>"
  rehydrate: "Dear <PERSON_1>, ..."      ->  "Dear Rahul, ..."

The Vault holds the placeholder <-> real value mapping for ONE request only.
It lives in memory, is never logged, never sent to the LLM, and is discarded
when the request finishes.
"""
import difflib
import re

from .detector import Finding

PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z_]*_\d+)>")
LOOSE_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][A-Za-z_]*_\d+>")


class Vault:
    def __init__(self):
        self._by_value: dict[tuple[str, str], str] = {}  # (type, value) -> placeholder
        self._by_placeholder: dict[str, str] = {}        # placeholder -> value
        self._counters: dict[str, int] = {}

    def placeholder_for(self, entity_type: str, value: str) -> str:
        """Same value always gets the same placeholder within a request, so the
        LLM can tell that two mentions refer to the same person."""
        key = (entity_type, value)
        if key not in self._by_value:
            n = self._counters.get(entity_type, 0) + 1
            self._counters[entity_type] = n
            ph = f"<{entity_type}_{n}>"
            self._by_value[key] = ph
            self._by_placeholder[ph] = value
        return self._by_value[key]

    def lookup(self, placeholder: str) -> str | None:
        return self._by_placeholder.get(placeholder)

    def __len__(self):
        return len(self._by_placeholder)

    def __contains__(self, placeholder: str) -> bool:
        return placeholder in self._by_placeholder

    def closest(self, placeholder: str) -> str | None:
        """The issued placeholder a garbled one was meant to be. The number
        must match exactly, so <PERSON_9> can never become <PERSON_1>."""
        wanted = placeholder.upper()
        number = wanted.rsplit("_", 1)[1]
        same_number = [p for p in self._by_placeholder if p.rsplit("_", 1)[1] == number]
        matches = difflib.get_close_matches(wanted, same_number, n=1, cutoff=0.8)
        return matches[0] if matches else None


def tokenize(text: str, findings: list[Finding], vault: Vault) -> str:
    # Replace from the end so earlier character offsets stay valid.
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        ph = vault.placeholder_for(f.entity_type, f.text)
        text = text[: f.start] + ph + text[f.end :]
    return text


def repair(text: str, vault: Vault) -> str:
    """Models garble placeholders ("<Person_1>", "<PERSION_1>", seen from
    gpt-4o-mini in 1 reply in 3). Restore the issued spelling."""
    return LOOSE_PLACEHOLDER_RE.sub(lambda m: vault.closest(m.group(0)) or m.group(0), text)


def rehydrate(text: str, vault: Vault) -> str:
    """Swap placeholders back to real values. A placeholder the model invented
    (not in the vault) is left as-is rather than guessed."""
    def swap(m: re.Match) -> str:
        real = vault.lookup(m.group(0))
        return real if real is not None else m.group(0)

    return PLACEHOLDER_RE.sub(swap, text)
