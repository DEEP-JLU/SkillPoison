from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.hts.counterfactual import outcome_from_trajectory
from core.hts.validation import render_trajectory
from core.legsa.evidence import ATTRIBUTION_SLOTS, LEGSA_PROMPT_FILES, AttributionEvidence


def load_legsa_prompts(prompts_dir: str | Path = Path("prompts") / "legsa") -> dict[str, str]:
    return {slot: (Path(prompts_dir) / filename).read_text(encoding="utf-8").strip() for slot, filename in LEGSA_PROMPT_FILES.items()}


def render_evidence_context(experience: dict, trajectory: dict, p_star: str, previous: dict[str, str] | None = None) -> str:
    previous = previous or {}
    blocks = [
        "Target behavior p*: " + p_star,
        "Task x_i:\n" + experience["task_text"],
        "Externally verified original outcome:\n" + str(experience["solution"]),
        "Original successful trajectory:\n" + render_trajectory(trajectory),
        "Validated counterfactual trajectory with p* replaced:\n" + render_trajectory(experience["counterfactual"]),
        "Original output z_i:\n" + outcome_from_trajectory(trajectory).text,
        "Counterfactual output z_tilde_i:\n" + outcome_from_trajectory(experience["counterfactual"]).text,
        "Counterfactual audit:\n" + str(experience["audits"]["counterfactual"]),
    ]
    for slot in ("e", "a"):
        if slot in previous:
            blocks.append(slot + ": " + previous[slot])
    return "\n\n".join(blocks)


@dataclass
class AttributionEvidenceGenerator:
    client: object
    prompts: dict[str, str]
    p_star: str

    def generate_attribution_evidence(self, experience: dict, trajectory: dict) -> AttributionEvidence:
        previous = {}
        for slot in ATTRIBUTION_SLOTS:
            previous[slot] = self.client.complete(system=self.prompts[slot], user=render_evidence_context(experience, trajectory, self.p_star, previous))
        return AttributionEvidence(**previous)

    @classmethod
    def from_paths(cls, client, p_star: str, prompts_dir: str | Path = Path("prompts") / "legsa") -> "AttributionEvidenceGenerator":
        return cls(client, load_legsa_prompts(prompts_dir), p_star)
