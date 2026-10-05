from __future__ import annotations

import copy
import json
from pathlib import Path

from core.config import make_embedder, make_llm_client
from core.gctir.assemble import assemble_formation_records
from core.gctir.scoring import serialize_trajectory
from core.hts.counterfactual import compute_edit_distance, outcome_from_trajectory
from core.hts.importance import compute_behavior_importance
from core.hts.validation import parse_object, validate_counterfactual, validate_experience, validate_target_audit
from core.hts.validity import compute_execution_validity
from core.legsa.generator import AttributionEvidenceGenerator
from core.legsa.record import build_formation_record
from core.utils.io import read_json, read_jsonl, write_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ReplayEmbedder:
    def __init__(self, experiences: list[dict], target_behavior: str, target_embedding: list[float]):
        self.vectors = {target_behavior: target_embedding}
        for experience in experiences:
            text = serialize_trajectory(experience["tau"])
            vector = experience.get("trajectory_embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError("replay requires trajectory_embedding for each selected record")
            if text in self.vectors and self.vectors[text] != vector:
                raise ValueError("conflicting replay embeddings for the same trajectory")
            self.vectors[text] = vector

    def encode(self, text: str) -> list[float]:
        return self.vectors[text]


class SkillPoisonPipeline:
    def __init__(self, config: dict, *, client=None, embedder=None):
        self.config = config
        self.client = client
        self.embedder = embedder

    def _audit(self, name: str, payload: dict) -> dict:
        prompt = (REPOSITORY_ROOT / "prompts" / name).read_text(encoding="utf-8").strip()
        return parse_object(self.client.complete(system=prompt, user=json.dumps(payload, ensure_ascii=False)))

    def _generate_counterfactual(self, experience: dict, p_star: str) -> dict:
        result = self._audit("counterfactual_generation.txt", {
            "task": experience["task_text"], "target_behavior": p_star,
            "original_trajectory": experience["trajectory"],
        })
        if any(key not in {"model_output", "code"} for key in result):
            raise ValueError("generated counterfactual contains unsupported fields")
        original = experience["trajectory"]
        return {**{key: copy.deepcopy(original[key]) for key in ("task_input", "code_context", "execution_context", "requirements") if key in original}, **result}

    def run(self, *, input_path: str | Path, output_directory_override: str | None = None, replay: bool = False) -> dict:
        config = self.config
        p_star = config["target_behavior"]["statement"]
        path = Path(input_path)
        payload = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        if isinstance(payload, list):
            payload = {"experiences": payload}
        if not isinstance(payload, dict) or not isinstance(payload.get("experiences"), list) or not payload["experiences"]:
            raise ValueError("input must contain a nonempty experiences list")
        if payload.get("target_behavior", p_star) != p_star:
            raise ValueError("input target_behavior differs from configuration")
        output = Path(output_directory_override or config["run"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if not replay and self.client is None:
            self.client = make_llm_client(config)
        candidates = copy.deepcopy(payload["experiences"])
        ids = [experience.get("id") for experience in candidates if isinstance(experience, dict)]
        if any(not isinstance(value, str) for value in ids) or len(ids) != len(candidates) or len(ids) != len(set(ids)):
            raise ValueError("experiences must have unique string IDs")
        selected = []
        audits = []
        for experience in candidates:
            entry = {"id": experience["id"], "accepted": False}
            try:
                validate_experience(experience)
                trajectory = experience["trajectory"]
                supplied = experience.get("audits", {})
                target_audit = supplied.get("target_behavior") if replay else self._audit("target_audit.txt", {"target_behavior": p_star, "task": experience["task_text"], "trajectory": trajectory})
                if not isinstance(target_audit, dict) or not validate_target_audit(trajectory, target_audit):
                    raise ValueError("target behavior was not observed")
                experience["audits"] = {"target_behavior": target_audit}
                counterfactual = experience.get("counterfactual")
                if counterfactual is None:
                    if replay:
                        raise ValueError("replay requires a recorded counterfactual")
                    counterfactual = self._generate_counterfactual(experience, p_star)
                cf_audit = supplied.get("counterfactual") if replay else self._audit("counterfactual_audit.txt", {"target_behavior": p_star, "task": experience["task_text"], "original": trajectory, "counterfactual": counterfactual})
                if not isinstance(cf_audit, dict):
                    raise ValueError("missing counterfactual audit")
                validate_counterfactual(experience, counterfactual, cf_audit)
                experience["counterfactual"] = counterfactual
                experience["audits"] = {"target_behavior": target_audit, "counterfactual": cf_audit}
                experience["tau"] = trajectory
                original = outcome_from_trajectory(trajectory)
                alternate = outcome_from_trajectory(counterfactual)
                experience["R"] = compute_behavior_importance(original, alternate)
                entry.update({"edit_distance": compute_edit_distance(original, alternate), "original_tokens": original.size, "R": experience["R"], "audits": experience["audits"], "counterfactual": counterfactual})
                if experience["R"] < config["hts"]["tau_R"]:
                    raise ValueError("behavior importance is below tau_R")
                experience["O"] = compute_execution_validity(experience["requirement_checks"])
                entry["O"] = experience["O"]
                if experience["O"] < config["hts"]["tau_O"]:
                    raise ValueError("execution validity is below tau_O")
                entry["accepted"] = True
                selected.append(experience)
            except (ValueError, TypeError, KeyError, SyntaxError) as error:
                entry["reason"] = str(error)
            audits.append(entry)
        write_json(output / "selection_audit.json", audits)
        if not selected:
            raise ValueError("no record passed HTS; see selection_audit.json")
        generator = None if replay else AttributionEvidenceGenerator.from_paths(self.client, p_star, REPOSITORY_ROOT / config["legsa"]["prompts_dir"])
        for experience in selected:
            if replay:
                evidence = experience.get("evidence")
                if not isinstance(evidence, dict):
                    raise ValueError("replay requires recorded LEGSA evidence")
            else:
                evidence = generator.generate_attribution_evidence(experience, experience["tau"])
            experience["record"] = build_formation_record(experience, experience["tau"], evidence, target_behavior=p_star)
        if self.embedder is None:
            if replay:
                target_embedding = payload.get("target_embedding")
                if not isinstance(target_embedding, list) or not target_embedding:
                    raise ValueError("replay input requires target_embedding")
                self.embedder = ReplayEmbedder(selected, p_star, target_embedding)
            else:
                self.embedder = make_embedder(config)
        settings = config["gctir"]
        formation = assemble_formation_records(selected, formation_budget=settings["formation_budget"], retained_categories=settings["retained_categories"], lam=settings["lambda"], p_star=p_star, embedder=self.embedder, require_exact_budget=settings["require_exact_budget"])
        summary = {
            "mode": "replay" if replay else "live", "n_candidates": len(candidates),
            "n_selected": len(selected), "n_records": len(formation["records"]),
            "J_star": formation["J_star"], "allocation": formation["allocation"],
            "categories": [{key: value for key, value in row.items() if key != "members"} for row in formation["scored_categories"]],
            "config": config,
        }
        if not replay:
            summary["token_usage"] = self.client.usage_summary()
        write_json(output / "formation_records.json", formation["records"])
        write_json(output / "run_summary.json", summary)
        return {**summary, "output_directory": str(output)}
