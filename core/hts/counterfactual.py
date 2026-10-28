from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class TextOutcome:
    text: str

    @property
    def size(self) -> int:
        return len(_tokens(self))


@dataclass(frozen=True)
class CodeOutcome:
    code: str

    @property
    def text(self) -> str:
        return self.code

    @property
    def size(self) -> int:
        return len(_tokens(self))


Outcome = TextOutcome | CodeOutcome


def _tokens(outcome: Outcome) -> list[str]:
    return _TOKEN_PATTERN.findall(outcome.text)


def compute_edit_distance(original: Outcome, counterfactual: Outcome) -> int:
    left, right = _tokens(original), _tokens(counterfactual)
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, x in enumerate(left, start=1):
        current = [i]
        for j, y in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def outcome_from_trajectory(trajectory: dict) -> Outcome:
    if trajectory.get("code"):
        return CodeOutcome(trajectory["code"])
    if trajectory.get("model_output"):
        return TextOutcome(trajectory["model_output"])
    raise ValueError("trajectory must contain nonempty code or model_output")
