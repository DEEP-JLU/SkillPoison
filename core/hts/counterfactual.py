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


def _longest_common_subsequence_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for x in left:
        current = [0]
        for index, y in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if x == y else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def compute_edit_distance(original: Outcome, counterfactual: Outcome) -> int:
    left, right = _tokens(original), _tokens(counterfactual)
    return max(len(left), len(right)) - _longest_common_subsequence_length(left, right)


def outcome_from_trajectory(trajectory: dict) -> Outcome:
    if trajectory.get("code"):
        return CodeOutcome(trajectory["code"])
    if trajectory.get("model_output"):
        return TextOutcome(trajectory["model_output"])
    raise ValueError("trajectory must contain nonempty code or model_output")
