from __future__ import annotations

import copy

from core.lega.evidence import AttributionEvidence, validate_attribution_evidence

FORMATION_RECORD_FIELDS = ("record_id", "category", "target_behavior", "task", "solution", "trajectory", "e", "a", "v", "kappa")


def build_formation_record(experience: dict, trajectory: dict, evidence: AttributionEvidence | dict, *, target_behavior: str) -> dict:
    attribution = validate_attribution_evidence(evidence)
    return {
        "record_id": experience["id"], "category": experience["category"],
        "target_behavior": target_behavior, "task": experience["task_text"],
        "solution": experience["solution"], "trajectory": copy.deepcopy(trajectory),
        **attribution.as_dict(),
    }
