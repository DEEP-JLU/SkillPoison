from __future__ import annotations

import ast
import json
import re

from core.hts.counterfactual import outcome_from_trajectory


def parse_object(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def render_trajectory(trajectory: dict) -> str:
    return "\n".join(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)}" for key, value in sorted(trajectory.items()))


def validate_output(trajectory: dict, output_format: str, allowed_labels: list[str] | None = None) -> None:
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory must be an object")
    outcome = outcome_from_trajectory(trajectory)
    if not isinstance(outcome.text, str) or not outcome.text.strip():
        raise ValueError("empty output")
    if trajectory.get("parse_success") is False or trajectory.get("parse_error"):
        raise ValueError("recorded output parsing failed")
    if trajectory.get("error_type") in {"syntax", "runtime_error", "no_code", "infrastructure", "timeout"}:
        raise ValueError("counterfactual has an execution or infrastructure error")
    if output_format == "json":
        value = parse_object(trajectory.get("model_output", ""))
        if allowed_labels is not None and value.get("label") not in allowed_labels:
            raise ValueError("output label is outside allowed_labels")
        if "predicted_label" in trajectory and trajectory["predicted_label"] != value.get("label"):
            raise ValueError("predicted_label differs from model_output")
    elif output_format == "python":
        if not trajectory.get("code"):
            raise ValueError("Python trajectories require a code field")
        ast.parse(trajectory["code"])
        if trajectory.get("model_output") and trajectory["model_output"].strip() != trajectory["code"].strip():
            raise ValueError("Python model_output differs from the recorded code")
    elif output_format != "text":
        raise ValueError("output_format must be json, python, or text")


def validate_experience(experience: dict) -> None:
    for field in ("id", "category", "task_text", "solution", "trajectory", "requirement_checks", "verified_success", "output_format"):
        if field not in experience:
            raise ValueError(f"missing experience field: {field}")
    for field in ("id", "category", "task_text", "solution"):
        if not isinstance(experience[field], str) or not experience[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    if experience["verified_success"] is not True:
        raise ValueError("formation inputs must be externally verified successful trajectories")
    checks = experience["requirement_checks"]
    if not isinstance(checks, list) or not checks or any(type(value) is not bool for value in checks):
        raise ValueError("requirement_checks must be a nonempty Boolean list")
    trajectory = experience["trajectory"]
    validate_output(trajectory, experience["output_format"], experience.get("allowed_labels"))
    if trajectory.get("task_input") != experience["task_text"]:
        raise ValueError("original task_input differs from task_text")
    if trajectory.get("passed") is False or trajectory.get("correct") is False:
        raise ValueError("original trajectory contradicts verified_success")


def _reject_negated_quote(source: str, quote: str) -> bool:
    negation = re.compile(r"\b(?:not|never|without|reject\w*|avoid\w*|didn't|doesn't|don't)\b", re.I)
    starts = [match.start() for match in re.finditer(re.escape(quote), source)]
    for start in starts:
        left = max(source.rfind(mark, 0, start) for mark in (";", "\n", ".")) + 1
        prefix = source[left:start]
        if not negation.search(prefix) and not negation.search(quote):
            return False
    return True


def validate_target_audit(trajectory: dict, audit: dict) -> bool:
    if type(audit.get("used")) is not bool:
        raise ValueError("target audit requires a Boolean used field")
    if not audit["used"]:
        return False
    quote = audit.get("evidence_quote")
    source = render_trajectory({key: trajectory[key] for key in ("model_output", "decision_basis", "evidence_used", "actions", "code") if key in trajectory})
    if not isinstance(quote, str) or not quote.strip() or quote not in source:
        raise ValueError("target audit must quote actual trajectory evidence")
    if _reject_negated_quote(source, quote):
        raise ValueError("target evidence is negated or rejected")
    if trajectory.get("code"):
        tree = ast.parse(trajectory["code"])
        segments = []
        def visit(node, inactive=False):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                for child in node.body:
                    visit(child, inactive or not bool(node.test.value))
                for child in node.orelse:
                    visit(child, inactive or bool(node.test.value))
                return
            if not inactive and isinstance(node, ast.Call):
                segments.append(ast.get_source_segment(trajectory["code"], node) or "")
            for child in ast.iter_child_nodes(node):
                visit(child, inactive)
        visit(tree)
        if not any(quote.strip() in segment for segment in segments):
            raise ValueError("Python target evidence must quote a call outside comments, strings, and constant-false branches")
    return True


def validate_counterfactual(experience: dict, counterfactual: dict, audit: dict) -> None:
    original = experience["trajectory"]
    validate_output(counterfactual, experience["output_format"], experience.get("allowed_labels"))
    for field in ("task_input", "code_context", "execution_context", "requirements"):
        if (field in original or field in counterfactual) and original.get(field) != counterfactual.get(field):
            raise ValueError(f"counterfactual changes preserved field: {field}")
    if experience["output_format"] == "json":
        left = parse_object(original["model_output"])
        right = parse_object(counterfactual["model_output"])
        if set(left) != set(right) or any(type(left[key]) is not type(right[key]) for key in left):
            raise ValueError("counterfactual changes the output schema")
    for field in ("task_compatible", "target_replaced", "other_conditions_unchanged"):
        if audit.get(field) is not True:
            raise ValueError(f"counterfactual audit rejected: {field}")
    if experience["output_format"] == "python":
        original_quote = experience.get("audits", {}).get("target_behavior", {}).get("evidence_quote")
        if original_quote:
            target_call = ast.dump(ast.parse(original_quote, mode="eval").body)
            if any(ast.dump(node) == target_call for node in ast.walk(ast.parse(counterfactual["code"])) if isinstance(node, ast.Call)):
                raise ValueError("counterfactual retains the quoted target call")
    quote = audit.get("alternative_evidence_quote")
    if not isinstance(quote, str) or not quote.strip() or quote not in render_trajectory(counterfactual):
        raise ValueError("counterfactual audit requires quoted evidence of the alternative")
    if outcome_from_trajectory(original).text == outcome_from_trajectory(counterfactual).text:
        raise ValueError("counterfactual does not change the recorded outcome")
