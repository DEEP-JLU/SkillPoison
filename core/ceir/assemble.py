from __future__ import annotations
from core.ceir.allocation import allocate_formation_budget
from core.ceir.categories import partition_by_category
from core.ceir.embedding import SemanticEmbedder
from core.ceir.scoring import score_categories, select_top_categories

def assemble_formation_records(selected_experiences: list[dict], *, formation_budget: int, retained_categories: int, lam: float, p_star: str, embedder: SemanticEmbedder, require_exact_budget: bool = False) -> dict:
    categories = partition_by_category(selected_experiences)
    scored_categories = score_categories(categories, lam=lam, p_star=p_star, embedder=embedder)
    retained = select_top_categories(scored_categories, retained_categories=retained_categories)
    selected, allocation = allocate_formation_budget(retained, formation_budget=formation_budget, require_exact_budget=require_exact_budget)
    return {'records': [experience.get('record') or {} for experience in selected], 'experiences': selected, 'scored_categories': scored_categories, 'J_star': [row['category'] for row in retained], 'allocation': allocation}
