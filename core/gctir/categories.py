from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class TaskCategory:
    key: str
    trigger: str
    members: list[dict] = field(default_factory=list)

    @property
    def n_j(self) -> int:
        return len(self.members)

def partition_by_category(selected_experiences: list[dict]) -> list[TaskCategory]:
    members_by_key: dict[str, list[dict]] = {}
    trigger_by_key: dict[str, str] = {}
    for experience in selected_experiences:
        key = str(experience.get('category') or 'unclassified')
        members_by_key.setdefault(key, []).append(experience)
        trigger_by_key.setdefault(key, str(experience.get('trigger') or key))
    return [TaskCategory(key=key, trigger=trigger_by_key[key], members=members) for key, members in sorted(members_by_key.items())]
