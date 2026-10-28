from __future__ import annotations
from core.ceir.categories import TaskCategory
from core.ceir.embedding import SemanticEmbedder, compute_similarity
TRAJECTORY_REPRESENTATION_FIELDS = ('task_input', 'actions', 'model_output', 'decision_basis', 'evidence_used', 'verification_result', 'passed', 'correct', 'code')

def serialize_trajectory(trajectory: dict | None) -> str | None:
    if not isinstance(trajectory, dict):
        return None
    parts = [f'{field}: {trajectory[field]}' for field in TRAJECTORY_REPRESENTATION_FIELDS if trajectory.get(field) not in (None, '')]
    return '\n'.join(parts) if parts else None

def compute_category_relevance(category: TaskCategory) -> float | None:
    if not category.members:
        return None
    values = [member.get('R') for member in category.members]
    if any((value is None for value in values)):
        return None
    return sum((float(value) for value in values)) / len(values)

def compute_trajectory_alignment(category: TaskCategory, p_star: str, embedder: SemanticEmbedder) -> float | None:
    if not p_star or not category.members:
        return None
    representations = [serialize_trajectory(member.get('tau')) for member in category.members]
    if any(not text for text in representations):
        raise ValueError("each selected trajectory must have an embedding representation")
    p_star_vector = embedder.encode(p_star)
    similarities = [compute_similarity(embedder.encode(text), p_star_vector) for text in representations]
    return sum(similarities) / len(similarities)

def score_categories(categories: list[TaskCategory], *, lam: float, p_star: str, embedder: SemanticEmbedder) -> list[dict]:
    scored = []
    for category in categories:
        relevance = compute_category_relevance(category)
        alignment = compute_trajectory_alignment(category, p_star, embedder)
        combined = None if relevance is None or alignment is None else relevance + lam * alignment
        scored.append({'category': category.key, 'trigger': category.trigger, 'n_j': category.n_j, 'S_j': relevance, 'A_j': alignment, 'U_j': combined, 'members': category.members})
    return scored

def select_top_categories(scored_categories: list[dict], *, retained_categories: int) -> list[dict]:
    defined = [row for row in scored_categories if row['U_j'] is not None]
    if type(retained_categories) is not int or not 1 <= retained_categories <= len(defined):
        raise ValueError('retained_categories must be between 1 and the number of scored categories')
    return sorted(defined, key=lambda row: (-row['U_j'], row['category']))[:retained_categories]
