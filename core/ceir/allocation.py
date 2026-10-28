from __future__ import annotations

def allocate_formation_budget(selected_categories: list[dict], *, formation_budget: int, require_exact_budget: bool = False) -> tuple[list[dict], dict]:
    if formation_budget < 0:
        raise ValueError('formation budget must be non-negative')
    if not selected_categories:
        if require_exact_budget and formation_budget:
            raise ValueError('insufficient retained records for the exact formation budget')
        return ([], {'formation_budget': formation_budget, 'b_j': {}, 'budget_used': 0})
    priority = sorted(range(len(selected_categories)), key=lambda index: (-selected_categories[index]['U_j'], selected_categories[index]['category']))
    capacities = [row['n_j'] for row in selected_categories]
    if require_exact_budget and sum(capacities) < formation_budget:
        raise ValueError(f'exact budget requires {formation_budget} records; retained categories contain {sum(capacities)}')
    base_share = formation_budget // len(selected_categories)
    allocations = [min(base_share, capacity) for capacity in capacities]
    remainder = formation_budget - sum(allocations)
    while remainder > 0:
        distributed = False
        for index in priority:
            if remainder == 0:
                break
            if allocations[index] < capacities[index]:
                allocations[index] += 1
                remainder -= 1
                distributed = True
        if not distributed:
            break
    selected_members: list[dict] = []
    for row, count in zip(selected_categories, allocations):
        ranked = sorted(row['members'], key=lambda member: -member['R'])
        selected_members.extend(ranked[:count])
    allocation = {'formation_budget': formation_budget, 'b_j': {row['category']: count for row, count in zip(selected_categories, allocations)}, 'budget_used': sum(allocations)}
    return (selected_members, allocation)
