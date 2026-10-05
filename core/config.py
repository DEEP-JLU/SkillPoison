from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import yaml

from core.gctir.embedding import DEFAULT_EMBEDDING_MODEL, SemanticEmbedder
from core.utils.llm import LLMClient

DEFAULT_CONFIG = {
    "run": {"output_dir": "outputs/run"},
    "model": {"name": "deepseek-v4-flash", "temperature": 0.0, "model_env": "MODEL_NAME", "base_url_env": "MODEL_BASE_URL", "api_key_env": "MODEL_API_KEY", "max_retries": 3},
    "target_behavior": {"statement": ""},
    "embedding": {"model": DEFAULT_EMBEDDING_MODEL, "normalize": True},
    "hts": {"tau_R": 0.5, "tau_O": 1.0},
    "legsa": {"prompts_dir": "prompts/legsa"},
    "gctir": {"lambda": 0.5, "retained_categories": 3, "formation_budget": 15, "require_exact_budget": True},
}


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path) -> dict:
    supplied = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("configuration must be a mapping")
    config = _merge(DEFAULT_CONFIG, supplied)
    if "seed" in config["run"]:
        raise ValueError("run.seed is not supported; core record selection uses fixed input order")
    statement = config["target_behavior"].get("statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("target_behavior.statement is required")
    for section, key in (("hts", "tau_R"), ("hts", "tau_O"), ("gctir", "lambda")):
        value = config[section][key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{section}.{key} must be in [0, 1]")
    for key in ("formation_budget", "retained_categories"):
        value = config["gctir"][key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"gctir.{key} must be a positive integer")
    for value in (config["gctir"]["require_exact_budget"], config["embedding"]["normalize"]):
        if type(value) is not bool:
            raise ValueError("budget and normalization switches must be Boolean")
    return config


def make_llm_client(config: dict) -> LLMClient:
    model = config["model"]
    return LLMClient.from_env(
        model=os.environ.get(model["model_env"]) or model["name"],
        api_key_env=model["api_key_env"], base_url_env=model["base_url_env"],
        temperature=float(model["temperature"]), max_retries=int(model["max_retries"]))


def make_embedder(config: dict) -> SemanticEmbedder:
    return SemanticEmbedder(model=config["embedding"]["model"], normalize=config["embedding"]["normalize"])
