from __future__ import annotations
from core.hts.counterfactual import Outcome, compute_edit_distance

def compute_behavior_importance(z_i: Outcome, z_tilde_i: Outcome) -> float:
    denominator = z_i.size
    if denominator <= 0:
        return 0.0
    return min(1.0, compute_edit_distance(z_i, z_tilde_i) / denominator)
