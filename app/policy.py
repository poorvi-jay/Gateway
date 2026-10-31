"""Loads policy.yaml: which entity types to scan for and what to do with each."""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_ACTIONS = {"tokenize", "block", "allow"}


@dataclass
class Policy:
    actions: dict = field(default_factory=dict)  # entity type -> action
    score_threshold: float = 0.5

    @property
    def scanned_entities(self) -> list[str]:
        """Entity types the detector should look for (everything not 'allow')."""
        return [e for e, a in self.actions.items() if a != "allow"]

    def action_for(self, entity_type: str) -> str:
        # Unknown types default to tokenize: safer than letting them through.
        return self.actions.get(entity_type, "tokenize")


def load_policy(path: str | Path) -> Policy:
    data = yaml.safe_load(Path(path).read_text())
    actions = data.get("entities", {})
    bad = {e: a for e, a in actions.items() if a not in VALID_ACTIONS}
    if bad:
        # Refuse to start with a broken policy rather than guess.
        raise ValueError(f"Invalid actions in policy: {bad}")
    return Policy(actions=actions, score_threshold=float(data.get("score_threshold", 0.5)))
