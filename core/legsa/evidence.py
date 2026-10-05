from __future__ import annotations
from dataclasses import asdict, dataclass
ATTRIBUTION_SLOTS = ('e', 'a', 'v', 'kappa')
LEGSA_PROMPT_FILES = {'e': 'task_evidence.txt', 'a': 'success_attribution.txt', 'v': 'outcome_consistency.txt', 'kappa': 'counterfactual.txt'}

@dataclass(frozen=True)
class AttributionEvidence:
    e: str
    a: str
    v: str
    kappa: str

    def as_dict(self) -> dict:
        return asdict(self)

def validate_attribution_evidence(evidence) -> AttributionEvidence:
    values = evidence.as_dict() if isinstance(evidence, AttributionEvidence) else dict(evidence)
    missing = [slot for slot in ATTRIBUTION_SLOTS if not isinstance(values.get(slot), str) or not values[slot].strip()]
    if missing:
        raise ValueError(f'attribution slot(s) {missing} are empty')
    return AttributionEvidence(**{slot: str(values[slot]).strip() for slot in ATTRIBUTION_SLOTS})
