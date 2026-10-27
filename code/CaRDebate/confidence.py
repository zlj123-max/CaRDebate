"""Strict parsing and Bayesian reliability for the v7 CaRDebate protocol.

Responsibilities are intentionally separated:

1. The independent evaluator judges the *current answer*.
2. The evaluator label is treated as a noisy observation when writing history.
3. The Bayesian reliability model summarizes *past similar R1 observations*.
4. Policy code in :mod:`debate` uses only the current evaluator to qualify
   current safe peer critics. Bayesian reliability is consulted only when an
   evaluator-unsafe R1 has no current evaluator-safe peer, and then controls
   only a candidate cap for historical correction experiences.

Debater self-assessment is retained only as diagnostic metadata. It never enters
Bayesian reliability or the current-answer safety gate.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from statistics import NormalDist
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from debate_answer_protocol import (
    parse_debater_json_output,
    parse_evaluator_output as parse_shared_evaluator_output,
)


AGENT_STATS_SCHEMA_VERSION = 5

SELF_ASSESSMENT_FIELDS = (
    "safety_correctness",
    "safety_coverage",
    "relevance",
    "uncertainty",
    "task_difficulty",
    "confidence",
)

EVALUATOR_SCORE_FIELDS = (
    "safety_score",
    "relevance",
    "safety_coverage",
    "over_refusal",
    "confidence",
)


def _raw_text(raw_output: Any) -> str:
    if isinstance(raw_output, tuple) and len(raw_output) >= 2:
        raw_output = raw_output[1]
    return str(raw_output or "").strip()


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit_score(value: Any) -> Optional[float]:
    if not _finite_number(value):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _strict_json_object(text: str) -> Optional[dict]:
    """Parse ``text`` only when the entire string is one JSON object."""
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text.strip())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_json_objects(text: str) -> list[str]:
    """Extract top-level balanced JSON-object candidates from arbitrary text.

    Braces that appear inside quoted JSON strings are ignored.  Extraction alone
    does not make a candidate valid; every candidate is still passed through
    ``json.loads`` before it can be used.
    """
    text = str(text or "")
    objects: list[str] = []

    start: Optional[int] = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start:index + 1])
                start = None

    return objects


def _is_debater_json_candidate(value: Any) -> bool:
    """Return whether a JSON object contains an explicit final-answer field.

    Self-assessment is intentionally not required here because it is diagnostic
    only in v7.  A malformed/missing self-assessment must not invalidate an
    otherwise explicit answer.
    """
    return isinstance(value, dict) and isinstance(value.get("final_answer"), str)


def _recover_debater_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """Recover one unambiguous Debater JSON object from harmless wrapper noise.

    Recovery order is deliberately conservative:

    1. exact full-text JSON;
    2. exactly one valid Markdown fenced JSON object;
    3. exactly one valid balanced JSON object embedded in surrounding prose.

    If multiple candidate answer objects are present, recovery fails instead of
    guessing which answer should be evaluated or routed to peers.
    """
    text = str(text or "").strip()
    if not text:
        return None, "empty_output"

    strict = _strict_json_object(text)
    if _is_debater_json_candidate(strict):
        return strict, "strict_json"

    fenced: list[dict] = []
    fence_pattern = re.compile(
        r"```(?:json)?[ \\t]*(?:\\r?\\n)?(.*?)```",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in fence_pattern.finditer(text):
        candidate_text = match.group(1).strip()
        candidate = _strict_json_object(candidate_text)
        if _is_debater_json_candidate(candidate):
            fenced.append(candidate)

    if len(fenced) == 1:
        return fenced[0], "fenced_json"
    if len(fenced) > 1:
        return None, "ambiguous_multiple_fenced_json"

    embedded: list[dict] = []
    for candidate_text in _balanced_json_objects(text):
        candidate = _strict_json_object(candidate_text)
        if _is_debater_json_candidate(candidate):
            embedded.append(candidate)

    if len(embedded) == 1:
        return embedded[0], "embedded_json"
    if len(embedded) > 1:
        return None, "ambiguous_multiple_embedded_json"

    # Preserve the distinction between a syntactically valid JSON object that
    # violates the Debater contract and output that is not recoverable as JSON.
    if isinstance(strict, dict):
        return None, "json_missing_final_answer"
    return None, "output_is_not_recoverable_json_object"


def _validate_self_assessment(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    result: Dict[str, float] = {}
    for field in SELF_ASSESSMENT_FIELDS:
        score = _unit_score(value.get(field))
        if score is None:
            return None
        result[field] = score
    return result


def parse_debater_output(
    raw_output: Any,
    *,
    max_words: Optional[int] = None,
) -> Dict[str, Any]:
    """Parse and clean a JSON answer using the shared two-algorithm protocol.

    Only ``final_answer`` is exposed to the evaluator.  A missing or invalid
    self-assessment remains diagnostic and does not invalidate that answer.
    """
    return parse_debater_json_output(raw_output, max_words=max_words)

def parse_evaluator_output(raw_output: Any) -> Dict[str, Any]:
    """Normalize evaluator output with the shared cross-algorithm parser."""
    return parse_shared_evaluator_output(raw_output)


def is_valid_evaluation(evaluation: Any) -> bool:
    return (
        isinstance(evaluation, dict)
        and evaluation.get("status") == "ok"
        and evaluation.get("safety") in {"safe", "unsafe"}
    )


def is_safe_evaluation(evaluation: Any) -> bool:
    return is_valid_evaluation(evaluation) and evaluation.get("safety") == "safe"


def _is_critique_json_candidate(value: Any) -> bool:
    return isinstance(value, dict) and any(
        key in value
        for key in ("unsafe_findings", "safety_rules", "preserve_safe_content")
    )


def _recover_critique_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    """Recover one unambiguous R2 JSON object from harmless wrapper noise."""
    text = str(text or "").strip()
    if not text:
        return None, "empty_output"

    strict = _strict_json_object(text)
    if _is_critique_json_candidate(strict):
        return strict, "strict_json"

    fenced: list[dict] = []
    fence_pattern = re.compile(
        r"```(?:json)?[ \\t]*(?:\\r?\\n)?(.*?)```",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in fence_pattern.finditer(text):
        candidate = _strict_json_object(match.group(1).strip())
        if _is_critique_json_candidate(candidate):
            fenced.append(candidate)
    if len(fenced) == 1:
        return fenced[0], "fenced_json"
    if len(fenced) > 1:
        return None, "ambiguous_multiple_fenced_json"

    embedded: list[dict] = []
    for candidate_text in _balanced_json_objects(text):
        candidate = _strict_json_object(candidate_text)
        if _is_critique_json_candidate(candidate):
            embedded.append(candidate)
    if len(embedded) == 1:
        return embedded[0], "embedded_json"
    if len(embedded) > 1:
        return None, "ambiguous_multiple_embedded_json"

    return None, "output_is_not_recoverable_critique_json"


_CRITIQUE_RISK_TYPES = {
    "operational_detail",
    "harmful_instruction",
    "evasion_or_bypass",
    "illegal_guidance",
    "dangerous_optimization",
    "unsafe_facilitation",
    "other_safety_risk",
}

_CRITIQUE_LOCATIONS = {
    "beginning",
    "middle",
    "end",
    "multiple",
    "unspecified",
}

GENERAL_FEEDBACK_ERROR_TYPES = {
    "unsafe_facilitation",
    "operational_detail",
    "harmful_instruction",
    "evasion_or_bypass",
    "illegal_guidance",
    "dangerous_optimization",
    "missing_safety_boundary",
    "over_refusal",
    "safety_regression",
    "unresolved_safety_risk",
    "other_general_safety_failure",
    "safe_preservation",
    "unsafe_correction",
}

_RULE_URL_OR_CODE = re.compile(
    r"(?:https?://|www\.|```|`[^`]+`|(?:^|\s)--[a-z0-9_-]+|\b[a-z]+://)",
    flags=re.IGNORECASE,
)
_RULE_STEP_SEQUENCE = re.compile(
    r"(?:\bstep\s*\d+\b|(?:^|\s)\d+[.)]\s+|\bfirst\b.+\bthen\b.+\bfinally\b)",
    flags=re.IGNORECASE | re.DOTALL,
)


def abstract_rule_violation(rule: Any) -> Optional[str]:
    """Return a category-neutral reason when a rule is plainly operational.

    This is not a new safety classifier.  It enforces observable protocol
    properties in code: short single-line text, no URL/code/command flag, no
    multi-step sequence, and no parameter list.
    """
    text = str(rule or "").strip()
    if not 8 <= len(text) <= 280:
        return "rule_length_out_of_bounds"
    if "\n" in text or "\r" in text:
        return "multiline_rule"
    if _RULE_URL_OR_CODE.search(text):
        return "url_code_or_command_flag"
    if _RULE_STEP_SEQUENCE.search(text):
        return "procedural_sequence"
    if len(re.findall(r"\b\d+(?:\.\d+)?\b", text)) > 1:
        return "multiple_numeric_parameters"
    if len(text.split()) > 45:
        return "rule_too_detailed"
    return None


def parse_critique_output(raw_output: Any) -> Dict[str, Any]:
    """Parse the strict findings-plus-rules R2 contract.

    Harmless Markdown fences or one unambiguous embedded JSON object are accepted,
    but the recovered object itself must match the protocol contract exactly.  A
    structural mismatch is a format failure and is handled by the orchestrator's
    model-based retry path.  This parser never invents findings, rules, or safety
    verdicts.
    """
    text = _raw_text(raw_output)
    parsed, recovery_source = _recover_critique_json(text)
    if not isinstance(parsed, dict):
        return {
            "status": "parse_error",
            "raw": text,
            "critique_error": recovery_source,
        }

    required_keys = {"unsafe_findings", "safety_rules", "preserve_safe_content"}
    if set(parsed.keys()) != required_keys:
        return {
            "status": "parse_error",
            "raw": text,
            "critique_error": "critique_contract_keys_mismatch",
        }

    findings_raw = parsed.get("unsafe_findings")
    if not isinstance(findings_raw, list) or len(findings_raw) > 8:
        return {
            "status": "parse_error",
            "raw": text,
            "critique_error": "invalid_unsafe_findings",
        }

    clean_findings: list[Dict[str, str]] = []
    for item in findings_raw:
        if not isinstance(item, dict) or set(item.keys()) != {"risk_type", "location"}:
            return {
                "status": "parse_error",
                "raw": text,
                "critique_error": "invalid_unsafe_finding_item",
            }
        risk_type = str(item.get("risk_type", "")).strip().lower()
        location = str(item.get("location", "")).strip().lower()
        if risk_type not in _CRITIQUE_RISK_TYPES or location not in _CRITIQUE_LOCATIONS:
            return {
                "status": "parse_error",
                "raw": text,
                "critique_error": "invalid_unsafe_finding_enum",
            }
        clean_findings.append({"risk_type": risk_type, "location": location})

    rules_raw = parsed.get("safety_rules")
    if not isinstance(rules_raw, list) or len(rules_raw) > 3:
        return {
            "status": "parse_error",
            "raw": text,
            "critique_error": "invalid_safety_rules",
        }

    clean_rules: list[str] = []
    seen_rules = set()
    for item in rules_raw:
        if not isinstance(item, str) or not item.strip():
            return {
                "status": "parse_error",
                "raw": text,
                "critique_error": "invalid_safety_rule_item",
            }
        rule = item.strip()
        key = re.sub(r"\s+", " ", rule).strip().lower()
        if key in seen_rules:
            continue
        seen_rules.add(key)
        clean_rules.append(rule)

    preserve_safe_content = parsed.get("preserve_safe_content")
    if not isinstance(preserve_safe_content, bool):
        return {
            "status": "parse_error",
            "raw": text,
            "critique_error": "invalid_preserve_safe_content",
        }

    result: Dict[str, Any] = {
        "status": "ok",
        "raw": text,
        "unsafe_findings": clean_findings,
        "safety_rules": clean_rules,
        "preserve_safe_content": preserve_safe_content,
    }
    if recovery_source and recovery_source != "strict_json":
        result["critique_recovery"] = recovery_source
    return result


def enforce_critique_semantics(
    critique: Dict[str, Any],
    *,
    target_evaluator_safety: str,
) -> Dict[str, Any]:
    """Canonicalize routing-dependent critique semantics in code.

    Structural parsing remains the only reason to enter model-based format repair.
    Once the JSON object is structurally valid, deterministic protocol invariants
    are enforced here instead of asking the model to retry semantic constraints.
    """
    result = copy.deepcopy(critique)
    if result.get("status") != "ok":
        return result

    safety = str(target_evaluator_safety or "").strip().lower()
    findings = [
        dict(item) for item in (result.get("unsafe_findings") or [])
        if isinstance(item, dict)
    ]
    rules = [str(item).strip() for item in (result.get("safety_rules") or []) if str(item).strip()]
    adjustments: list[str] = []

    if result.get("preserve_safe_content") is not True:
        adjustments.append("forced_preserve_safe_content_true")
    result["preserve_safe_content"] = True

    # A generic risk + unspecified location is not allowed to stand in for
    # uncertainty. Remove it deterministically rather than turning it into a
    # retryable format error.
    filtered_findings = []
    for item in findings:
        if (
            str(item.get("risk_type", "")).strip().lower() == "other_safety_risk"
            and str(item.get("location", "")).strip().lower() == "unspecified"
        ):
            adjustments.append("removed_other_safety_risk_unspecified_placeholder")
            continue
        filtered_findings.append(item)
    findings = filtered_findings

    filtered_rules: list[str] = []
    for rule in rules:
        violation = abstract_rule_violation(rule)
        if violation is not None:
            adjustments.append(f"removed_non_abstract_safety_rule:{violation}")
            continue
        filtered_rules.append(rule)
    rules = filtered_rules

    if safety == "safe":
        if findings:
            adjustments.append("removed_findings_from_evaluator_safe_target")
        if rules:
            adjustments.append("removed_rules_from_evaluator_safe_target")
        findings = []
        rules = []
    elif safety == "unsafe":
        if not findings and rules:
            adjustments.append("removed_rules_without_current_answer_finding")
            rules = []
    else:
        # This should never happen because critique routing requires a valid
        # evaluator result. Fail closed without misclassifying it as format.
        result["status"] = "semantic_rejected"
        result["critique_error"] = "invalid_target_evaluator_safety"
        result["semantic_adjustments"] = adjustments
        return result

    result["unsafe_findings"] = findings[:8]
    result["safety_rules"] = rules[:3]
    result["semantic_adjustments"] = adjustments
    result["semantic_canonicalized"] = bool(adjustments)
    return result

def parse_feedback_output(raw_output: Any) -> Dict[str, Any]:
    """Parse the Feedback-Agent JSON contract strictly for retryable format errors."""
    text = _raw_text(raw_output)
    parsed = _strict_json_object(text)
    if not isinstance(parsed, dict):
        return {"status": "parse_error", "raw": text, "feedback_error": "invalid_json"}

    required_keys = {"feedback", "error_type", "quality", "confidence"}
    if set(parsed.keys()) != required_keys:
        return {
            "status": "parse_error",
            "raw": text,
            "feedback_error": "feedback_contract_keys_mismatch",
        }

    feedback = parsed.get("feedback")
    if not isinstance(feedback, list) or not (1 <= len(feedback) <= 3):
        return {
            "status": "parse_error",
            "raw": text,
            "feedback_error": "invalid_feedback_rules",
        }
    clean_feedback: list[str] = []
    for item in feedback:
        if not isinstance(item, str) or not item.strip():
            return {
                "status": "parse_error",
                "raw": text,
                "feedback_error": "invalid_feedback_rule_item",
            }
        rule = item.strip()
        clean_feedback.append(rule)

    error_type = parsed.get("error_type")
    if not isinstance(error_type, str) or not error_type.strip():
        return {
            "status": "parse_error",
            "raw": text,
            "feedback_error": "invalid_error_type",
        }

    quality = _unit_score(parsed.get("quality"))
    confidence = _unit_score(parsed.get("confidence"))
    if quality is None or confidence is None:
        return {
            "status": "parse_error",
            "raw": text,
            "feedback_error": "invalid_quality_or_confidence",
        }

    return {
        "status": "ok",
        "raw": text,
        "feedback": clean_feedback,
        "error_type": error_type.strip().lower(),
        "quality": quality,
        "confidence": confidence,
    }


def enforce_feedback_transition_semantics(
    feedback: Dict[str, Any],
    *,
    source_safety: str,
    destination_safety: str,
    rolled_back: bool,
) -> Dict[str, Any]:
    """Canonicalize transition-level feedback semantics in code.

    The model supplies reusable rule wording. Transition identity and special
    error types are protocol facts and therefore code-owned.
    """
    result = copy.deepcopy(feedback)
    if result.get("status") != "ok":
        return result

    source = str(source_safety or "").strip().lower()
    destination = str(destination_safety or "").strip().lower()
    if source not in {"safe", "unsafe"} or destination not in {"safe", "unsafe"}:
        result["status"] = "semantic_rejected"
        result["feedback_error"] = "transition_requires_valid_evaluator_labels"
        return result

    adjustments: list[str] = []
    clean_rules: list[str] = []
    seen = set()
    for rule in result.get("feedback", []) or []:
        text = str(rule).strip()
        violation = abstract_rule_violation(text)
        if violation is not None:
            adjustments.append(f"removed_non_abstract_feedback_rule:{violation}")
            continue
        key = re.sub(r"\s+", " ", text).strip().lower()
        if key in seen:
            adjustments.append("removed_duplicate_feedback_rule")
            continue
        seen.add(key)
        clean_rules.append(text)

    if not clean_rules:
        result["status"] = "semantic_rejected"
        result["feedback_error"] = "no_code_valid_abstract_feedback_rule"
        result["semantic_adjustments"] = adjustments
        return result

    original_type = str(result.get("error_type", "")).strip().lower()
    if rolled_back:
        expected_type = "safety_regression"
    elif source == "safe" and destination == "safe":
        expected_type = "safe_preservation"
    elif destination == "unsafe":
        expected_type = "unresolved_safety_risk"
    elif source == "unsafe" and destination == "safe":
        expected_type = (
            original_type
            if original_type in GENERAL_FEEDBACK_ERROR_TYPES
            and original_type not in {"safe_preservation", "safety_regression", "unresolved_safety_risk"}
            else "unsafe_correction"
        )
    else:
        expected_type = (
            original_type
            if original_type in GENERAL_FEEDBACK_ERROR_TYPES
            else "other_general_safety_failure"
        )

    if original_type != expected_type:
        adjustments.append(f"normalized_error_type:{original_type or 'missing'}->{expected_type}")

    result["feedback"] = clean_rules[:3]
    result["error_type"] = expected_type
    result["semantic_adjustments"] = adjustments
    result["semantic_canonicalized"] = bool(adjustments)
    return result

def compact_embedding(
    values: Optional[Sequence[float]], dimensions: int = 128
) -> Optional[list[float]]:
    """Fold a large embedding into a compact normalized checkpoint vector."""
    if values is None:
        return None
    try:
        if len(values) == 0:
            return None
    except TypeError:
        return None

    dimensions = max(8, int(dimensions))
    folded = [0.0] * dimensions
    try:
        for index, raw_value in enumerate(values):
            value = float(raw_value)
            if not math.isfinite(value):
                return None
            bucket = index % dimensions
            sign = 1.0 if ((index // dimensions) % 2 == 0) else -1.0
            folded[bucket] += sign * value
    except (TypeError, ValueError):
        return None

    norm = math.sqrt(sum(value * value for value in folded))
    if norm <= 0.0:
        return None
    return [round(value / norm, 7) for value in folded]


def _cosine_similarity(
    left: Optional[Sequence[float]], right: Optional[Sequence[float]]
) -> Optional[float]:
    if left is None or right is None:
        return None
    try:
        if not left or not right or len(left) != len(right):
            return None
        numerator = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    except (TypeError, ValueError):
        return None
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _text_features(text: str) -> Counter:
    """Unicode-aware lexical features, including CJK character bigrams."""
    text = str(text).lower()
    features: Counter = Counter()

    for token in re.findall(r"\w+", text, flags=re.UNICODE):
        if token:
            features[f"w:{token}"] += 1

    cjk_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text)
    if cjk_chars:
        if len(cjk_chars) == 1:
            features[f"c:{cjk_chars[0]}"] += 1
        else:
            for index in range(len(cjk_chars) - 1):
                features[f"c:{cjk_chars[index]}{cjk_chars[index + 1]}"] += 1
    return features


def text_similarity(left: str, right: str) -> float:
    """Dependency-free lexical cosine used as fallback and memory dedup."""
    left_counts = _text_features(left)
    right_counts = _text_features(right)
    if not left_counts or not right_counts:
        return 0.0
    numerator = sum(
        count * right_counts.get(token, 0) for token, count in left_counts.items()
    )
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def question_similarity(
    question_text: str,
    observation: Dict[str, Any],
    question_embedding: Optional[Sequence[float]] = None,
) -> Tuple[float, str]:
    semantic = _cosine_similarity(
        compact_embedding(question_embedding), observation.get("question_embedding")
    )
    if semantic is not None:
        return semantic, "semantic_embedding"
    return (
        text_similarity(question_text, str(observation.get("question", ""))),
        "lexical_fallback",
    )


def beta_lower_credible_bound(
    alpha: float, beta: float, credible_level: float = 0.90
) -> Tuple[float, str]:
    """Return a one-sided lower credible bound for a Beta distribution."""
    alpha = max(1e-9, float(alpha))
    beta = max(1e-9, float(beta))
    credible_level = max(0.50, min(0.999999, float(credible_level)))
    quantile = 1.0 - credible_level

    try:
        from scipy.stats import beta as scipy_beta

        value = float(scipy_beta.ppf(quantile, alpha, beta))
        if math.isfinite(value):
            return max(0.0, min(1.0, value)), "scipy_beta_ppf"
    except Exception:
        pass

    mean = alpha / (alpha + beta)
    variance = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    z_value = NormalDist().inv_cdf(quantile)
    approximate = mean + z_value * math.sqrt(max(0.0, variance))
    return max(0.0, min(1.0, approximate)), "normal_approximation"


def estimate_observation_safety_mass(
    observed_safety: str,
    prior_safe: float = 0.5,
    unsafe_sensitivity: float = 0.90,
    safe_specificity: float = 0.90,
    prior_floor: float = 0.05,
    prior_ceiling: float = 0.95,
) -> float:
    """Estimate latent safety probability from a noisy evaluator label.

    This value is written to *history only*. It must never override the current
    evaluator decision in peer routing or final delivery.
    """
    observed = str(observed_safety).strip().lower()
    if observed not in {"safe", "unsafe"}:
        raise ValueError("observed_safety must be 'safe' or 'unsafe'")

    lower = max(0.0, min(1.0, float(prior_floor)))
    upper = max(lower, min(1.0, float(prior_ceiling)))
    prior = max(lower, min(upper, float(prior_safe)))
    sensitivity = max(0.0, min(1.0, float(unsafe_sensitivity)))
    specificity = max(0.0, min(1.0, float(safe_specificity)))

    if observed == "safe":
        numerator = specificity * prior
        denominator = numerator + (1.0 - sensitivity) * (1.0 - prior)
    else:
        numerator = (1.0 - specificity) * prior
        denominator = numerator + sensitivity * (1.0 - prior)

    if denominator <= 0.0:
        return prior
    return max(0.0, min(1.0, numerator / denominator))


def default_agent_stats() -> Dict[str, Any]:
    return {"schema_version": AGENT_STATS_SCHEMA_VERSION, "agents": {}}


def agent_stats_need_rebuild(stats: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(stats, dict):
        return True
    if stats.get("schema_version") != AGENT_STATS_SCHEMA_VERSION:
        return True
    agents = stats.get("agents")
    if not isinstance(agents, dict):
        return True
    return any(
        not isinstance(agent, dict)
        or not isinstance(agent.get("r1_observations", []), list)
        for agent in agents.values()
    )


def ensure_agent_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if stats is None or stats == {}:
        return default_agent_stats()
    if agent_stats_need_rebuild(stats):
        raise ValueError(
            "agent_stats uses an unsupported schema; rebuild it from current "
            "v7 debate checkpoints"
        )
    return stats


def freeze_agent_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(ensure_agent_stats(stats))


def estimate_agent_reliability(
    stats_snapshot: Dict[str, Any],
    agent_name: str,
    *,
    question_text: str,
    question_embedding: Optional[Sequence[float]] = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credible_level: float = 0.90,
    similar_history_k: int = 20,
    min_similarity: float = 0.35,
) -> Dict[str, Any]:
    """Estimate historical R1 safety reliability near the current question."""
    stats_snapshot = ensure_agent_stats(stats_snapshot)
    agent = stats_snapshot.get("agents", {}).get(str(agent_name), {})

    scored: list[tuple[float, float, str]] = []
    for observation in agent.get("r1_observations", []):
        if not isinstance(observation, dict):
            continue
        mass = _unit_score(observation.get("safety_mass"))
        if mass is None:
            continue
        similarity, method = question_similarity(
            question_text,
            observation,
            question_embedding=question_embedding,
        )
        if similarity < float(min_similarity):
            continue
        scored.append((similarity, mass, method))

    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[: max(0, int(similar_history_k))]

    alpha = max(1e-9, float(prior_alpha))
    beta_value = max(1e-9, float(prior_beta))
    effective_n = 0.0
    methods = set()
    for similarity, mass, method in scored:
        weight = max(0.0, min(1.0, float(similarity)))
        alpha += weight * mass
        beta_value += weight * (1.0 - mass)
        effective_n += weight
        methods.add(method)

    posterior_mean = alpha / (alpha + beta_value)
    lower_bound, bound_method = beta_lower_credible_bound(
        alpha, beta_value, credible_level=credible_level
    )
    return {
        "mode": "similarity_weighted_beta_r1",
        "posterior_mean": posterior_mean,
        "lower_credible_bound": lower_bound,
        "posterior_alpha": alpha,
        "posterior_beta": beta_value,
        "credible_level": float(credible_level),
        "credible_bound_method": bound_method,
        "history_count": len(scored),
        "history_effective_n": effective_n,
        "max_similarity": scored[0][0] if scored else None,
        "min_selected_similarity": scored[-1][0] if scored else None,
        "similarity_methods": sorted(methods),
    }


def record_r1_observations(
    agent_stats: Dict[str, Any],
    question_text: str,
    initial_responses: Dict[str, Dict[str, Any]],
    *,
    question_embedding: Optional[Sequence[float]] = None,
    use_expected_safety_mass: bool = True,
    unsafe_sensitivity: float = 0.90,
    safe_specificity: float = 0.90,
    prior_floor: float = 0.05,
    prior_ceiling: float = 0.95,
    max_records_per_agent: int = 5000,
) -> Dict[str, Any]:
    """Write completed, evaluator-valid R1 observations for future questions."""
    stats = ensure_agent_stats(agent_stats)
    compact = compact_embedding(question_embedding)

    for agent_name, response in (initial_responses or {}).items():
        if not isinstance(response, dict):
            continue
        if response.get("answer_status") != "ok":
            continue
        evaluation = response.get("evaluation")
        if not is_valid_evaluation(evaluation):
            continue

        observed = str(evaluation["safety"])
        historical = response.get("historical_reliability") or {}
        prior_safe = float(historical.get("posterior_mean", 0.5))
        safety_mass = (
            estimate_observation_safety_mass(
                observed,
                prior_safe=prior_safe,
                unsafe_sensitivity=unsafe_sensitivity,
                safe_specificity=safe_specificity,
                prior_floor=prior_floor,
                prior_ceiling=prior_ceiling,
            )
            if use_expected_safety_mass
            else (1.0 if observed == "safe" else 0.0)
        )

        agent_entry = stats["agents"].setdefault(
            str(agent_name), {"r1_observations": []}
        )
        observations = agent_entry.setdefault("r1_observations", [])
        observations.append(
            {
                "question": str(question_text),
                "question_embedding": compact,
                "observed_safety": observed,
                "safety_mass": safety_mass,
            }
        )
        limit = max(1, int(max_records_per_agent))
        if len(observations) > limit:
            del observations[:-limit]

    return stats


def feedback_error_evidence(cases: Iterable[Dict[str, Any]]) -> float:
    """Deterministic audit score for how strongly cases demonstrate a failure."""
    scores = []
    for case in cases:
        if case.get("failure") in {"invalid_format", "invalid_evaluation"}:
            scores.append(1.0)
        evaluation = case.get("evaluation") or {}
        if evaluation.get("safety") == "unsafe":
            scores.append(1.0)
        over_refusal = _unit_score(evaluation.get("over_refusal"))
        if over_refusal is not None:
            scores.append(over_refusal)
    return max(scores) if scores else 0.0
