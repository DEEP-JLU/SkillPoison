from __future__ import annotations

def compute_execution_validity(requirement_checks: list[bool]) -> float:
    if not requirement_checks:
        return 0.0
    return sum((1 for satisfied in requirement_checks if satisfied)) / len(requirement_checks)
