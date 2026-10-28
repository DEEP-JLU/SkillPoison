from __future__ import annotations

from core.hts.counterfactual import Outcome, compute_edit_distance


def compute_behavior_importance(z_i: Outcome, z_tilde_i: Outcome) -> float:
    return compute_edit_distance(z_i, z_tilde_i) / max(1, z_i.size, z_tilde_i.size)
