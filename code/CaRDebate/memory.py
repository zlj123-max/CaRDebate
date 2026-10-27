"""Long-term memory backends for the v7 CaRDebate protocol.

Short-term information lives in the EpisodeState defined in debate.py. This file
contains only cross-question long-term memory backends. Models receive only the
retrieved correction-rule texts selected by the orchestrator.

Learned feedback uses a question-keyed retrieval schema:

    retrieval key  = historical source question
    returned value = reusable feedback correction rule

This avoids comparing the current question directly with an abstract correction
rule. Learned feedback is eligible for corrective retrieval only when its source
trajectory ended with an evaluator-valid SAFE final answer. Category-aware retrieval
can request same-category or cross-category successful histories explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_RANKING_WEIGHTS = {
    "relevance": 0.70,
    "support": 0.20,
    "outcome_priority": 0.10,
}


def _unit_metadata_score(
    metadata: Optional[dict], key: str, default: float = 1.0
) -> float:
    try:
        value = float((metadata or {}).get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(1.0, value))


def _feedback_support(metadata: Optional[dict]) -> float:
    metadata = metadata or {}
    if "feedback_support" in metadata:
        return _unit_metadata_score(metadata, "feedback_support", 0.0)
    quality = _unit_metadata_score(metadata, "quality", 0.0)
    confidence = _unit_metadata_score(metadata, "confidence", 0.0)
    return 0.5 * quality + 0.5 * confidence


def _verified_final_safe(metadata: Optional[dict]) -> bool:
    metadata = metadata or {}
    if str(metadata.get("source_final_safety", "")).strip().lower() == "safe":
        return bool(metadata.get("source_final_safety_valid", True))
    return bool(metadata.get("verified_safe", False))


def _transition_priority(metadata: Optional[dict]) -> float:
    transition = str((metadata or {}).get("source_transition", "")).strip().lower()
    if transition == "unsafe_to_safe":
        return 1.0
    if transition == "safe_to_safe":
        return 0.60
    return 0.0


def _category_matches(metadata: Optional[dict], category: Optional[str]) -> bool:
    if category is None:
        return True
    value = (metadata or {}).get("category")
    return value is not None and str(value) == str(category)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _relevance_summary(values: Iterable[float]) -> Dict[str, float]:
    vals = [max(0.0, min(1.0, float(v))) for v in values]
    if not vals:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": _mean(vals),
    }


def _record(
    text: str,
    metadata: Optional[dict],
    relevance: float,
    relevance_method: str,
    *,
    retrieval_text: Optional[str] = None,
    ranking_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    relevance = max(0.0, min(1.0, float(relevance)))
    support = _feedback_support(metadata)
    outcome_priority = _transition_priority(metadata)
    # Relevance remains primary; support and a modest successful-correction bonus
    # prefer histories that actually moved UNSAFE -> SAFE.
    weights = dict(DEFAULT_RANKING_WEIGHTS)
    if isinstance(ranking_weights, dict):
        weights.update(
            {
                str(key): float(value)
                for key, value in ranking_weights.items()
                if key in DEFAULT_RANKING_WEIGHTS
            }
        )
    ranking_score = (
        weights["relevance"] * relevance
        + weights["support"] * support
        + weights["outcome_priority"] * outcome_priority
    )
    return {
        "text": str(text),
        "retrieval_text": str(retrieval_text or metadata.get("source_question") or ""),
        "relevance": relevance,
        "relevance_method": relevance_method,
        "feedback_support": support,
        "quality": _unit_metadata_score(metadata, "quality", 0.0),
        "confidence": _unit_metadata_score(metadata, "confidence", 0.0),
        "ranking_score": max(0.0, min(1.0, ranking_score)),
        "outcome_priority": outcome_priority,
        "source_question_id": metadata.get("source_question_id"),
        "source_agent": metadata.get("source_agent"),
        "source_category": metadata.get("category"),
        "source_final_safety": metadata.get("source_final_safety"),
        "source_transition": metadata.get("source_transition"),
        "metadata": metadata,
    }


def _normalize_seed_memories(
    memories: Optional[Iterable[Any]], category: Optional[str]
) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    if memories is None:
        items: Iterable[Any] = []
    elif isinstance(memories, (str, dict)):
        items = [memories]
    else:
        items = memories
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text", item.get("memory", ""))).strip()
            metadata = dict(item.get("metadata") or {})
        else:
            text = str(item).strip()
            metadata = {}
        if not text:
            continue
        metadata.setdefault("memory_type", "seed")
        metadata.setdefault("feedback_support", 1.0)
        metadata.setdefault("quality", 1.0)
        metadata.setdefault("confidence", 1.0)
        if category is not None:
            metadata.setdefault("category", str(category))
        result.append({"text": text, "metadata": metadata})
    return result



class LongTermMemory:
    """Array-backed successful-correction episode memory.

    Learned cross-question history is deliberately narrow:

        evaluator-verified UNSAFE -> SAFE correction episode
            -> reusable critique/correction rules

    One historical question is one episode.  Rules live under the episode rather
    than competing as separate retrieval documents.
    """

    RETRIEVAL_SCHEMA = "unsafe_to_safe_episode_rule_v1"

    def __init__(
        self, description: str = "Long-Term Memory", *, allow_all_history: bool = False
    ) -> None:
        self.description = str(description)
        self.allow_all_history = bool(allow_all_history)
        if self.allow_all_history:
            self.RETRIEVAL_SCHEMA = "all_transition_episode_rule_v1"
        self.episode_records: Dict[str, Dict[str, Any]] = {}
        # Kept for runner/export compatibility: this is a flattened view of rules.
        self.feedback_records: list[Dict[str, Any]] = []
        self.seed_records: list[Dict[str, Any]] = []

    def set_seed_memories(
        self, memories: Optional[Iterable[Any]], category: Optional[str] = None
    ) -> None:
        """Replace only question-local seed memories; learned episodes are untouched."""
        self.seed_records = _normalize_seed_memories(memories, category)

    @staticmethod
    def _episode_id(
        source_question: str,
        metadata: Optional[dict] = None,
    ) -> str:
        metadata = dict(metadata or {})
        for key in ("episode_id", "source_question_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return "episode:" + hashlib.sha256(
            str(source_question).encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _is_successful_correction_metadata(metadata: Optional[dict]) -> bool:
        metadata = dict(metadata or {})
        transition = str(metadata.get("source_transition") or "").strip().lower()
        final_safety = str(metadata.get("source_final_safety") or "").strip().lower()
        final_valid = bool(metadata.get("source_final_safety_valid", True))
        outcome_verified = bool(metadata.get("source_outcome_verified", True))
        return (
            transition == "unsafe_to_safe"
            and final_safety == "safe"
            and final_valid
            and outcome_verified
        )

    def _eligible_history_metadata(self, metadata: Optional[dict]) -> bool:
        """Storage/retrieval gate; keeps baseline correction-only behavior by default."""
        if not self.allow_all_history:
            return self._is_successful_correction_metadata(metadata)
        meta = dict(metadata or {})
        return bool(
            meta.get("source_final_safety_valid", False)
            and meta.get("source_outcome_verified", False)
            and str(meta.get("source_transition") or "").lower() in {
                "unsafe_to_safe", "safe_to_safe", "unsafe_to_unsafe",
                "safe_to_unsafe", "all_transitions",
            }
        )

    def _normalize_episode_rule(
        self,
        item: Any,
        *,
        episode_id: str,
        source_question: str,
        episode_metadata: dict,
    ) -> Optional[Dict[str, Any]]:
        if isinstance(item, tuple) and len(item) >= 2:
            text = str(item[0]).strip()
            metadata = dict(item[1] or {})
        elif isinstance(item, dict):
            text = str(item.get("text", item.get("rule", ""))).strip()
            metadata = dict(item.get("metadata") or {})
            if not metadata:
                metadata = {
                    str(k): v
                    for k, v in item.items()
                    if k not in {"text", "rule", "metadata"}
                }
        else:
            text = str(item).strip()
            metadata = {}

        if not text:
            return None

        merged = dict(episode_metadata)
        merged.update(metadata)
        merged["episode_id"] = episode_id
        merged["source_question"] = source_question
        merged["source_question_id"] = (
            merged.get("source_question_id") or episode_id
        )
        merged["memory_type"] = "feedback_rule"
        merged["feedback_rule"] = text
        merged["retrieval_schema"] = self.RETRIEVAL_SCHEMA
        if not self.allow_all_history:
            # Validate the original rule provenance before canonicalizing fields.
            # Never relabel other transition types as successful corrections.
            if not self._is_successful_correction_metadata(merged):
                return None
            merged["source_transition"] = "unsafe_to_safe"
            merged["source_final_safety"] = "safe"
            merged["source_final_safety_valid"] = True
            merged["source_outcome_verified"] = True

        if not self._eligible_history_metadata(merged):
            return None

        return {
            "text": text,
            "retrieval_text": source_question,
            "metadata": merged,
        }

    @staticmethod
    def _rule_sort_key(record: Dict[str, Any]) -> tuple[float, float, float]:
        metadata = dict(record.get("metadata") or {})
        return (
            _feedback_support(metadata),
            _unit_metadata_score(metadata, "quality", 0.0),
            _unit_metadata_score(metadata, "confidence", 0.0),
        )

    def _build_episode_record(
        self,
        *,
        episode_id: str,
        source_question: str,
        rules: Iterable[Any],
        metadata: Optional[dict] = None,
        max_rules: Optional[int] = None,
    ) -> Dict[str, Any]:
        episode_id = str(episode_id).strip()
        source_question = str(source_question).strip()
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if not source_question:
            raise ValueError("source_question must not be empty")

        episode_metadata = dict(metadata or {})
        episode_metadata["episode_id"] = episode_id
        episode_metadata["source_question"] = source_question
        episode_metadata["source_question_id"] = (
            episode_metadata.get("source_question_id") or episode_id
        )
        episode_metadata["memory_type"] = "feedback_episode"
        episode_metadata["retrieval_schema"] = self.RETRIEVAL_SCHEMA
        if self.allow_all_history:
            episode_metadata["source_transition"] = "all_transitions"
            episode_metadata["source_final_safety"] = "mixed"
            episode_metadata["source_final_safety_valid"] = True
            episode_metadata["source_outcome_verified"] = True
        else:
            episode_metadata["source_transition"] = "unsafe_to_safe"
            episode_metadata["source_final_safety"] = "safe"
            episode_metadata["source_final_safety_valid"] = True
            episode_metadata["source_outcome_verified"] = True

        if not self._eligible_history_metadata(episode_metadata):
            raise ValueError("Long-term memory episode does not meet the configured storage policy")

        normalized: list[Dict[str, Any]] = []
        existing = self.episode_records.get(episode_id)
        if isinstance(existing, dict):
            normalized.extend(
                dict(item)
                for item in (existing.get("rules") or [])
                if isinstance(item, dict)
            )

        for item in rules:
            record = self._normalize_episode_rule(
                item,
                episode_id=episode_id,
                source_question=source_question,
                episode_metadata=episode_metadata,
            )
            if record is not None:
                normalized.append(record)

        # Exact textual dedup inside one episode.  Semantic dedup is performed in
        # debate.py, where the existing text_similarity helper is already available.
        best_by_text: Dict[str, Dict[str, Any]] = {}
        for record in normalized:
            text = str(record.get("text") or "").strip()
            if not text:
                continue
            key = " ".join(text.lower().split())
            previous = best_by_text.get(key)
            if previous is None or self._rule_sort_key(record) > self._rule_sort_key(previous):
                best_by_text[key] = record

        ordered = sorted(
            best_by_text.values(),
            key=self._rule_sort_key,
            reverse=True,
        )
        if max_rules is not None:
            ordered = ordered[: max(0, int(max_rules))]

        correction_keys = {
            (
                str((item.get("metadata") or {}).get("source_agent") or ""),
                str((item.get("metadata") or {}).get("source_stage") or ""),
                str((item.get("metadata") or {}).get("destination_stage") or ""),
            )
            for item in ordered
        }
        correction_keys.discard(("", "", ""))

        episode_metadata["rule_count"] = len(ordered)
        episode_metadata["successful_correction_count"] = max(
            int(episode_metadata.get("successful_correction_count", 0) or 0),
            len(correction_keys),
        )

        return {
            "episode_id": episode_id,
            "source_question": source_question,
            "metadata": episode_metadata,
            "rules": ordered,
        }

    def _rebuild_feedback_records(self) -> None:
        flattened: list[Dict[str, Any]] = []
        for episode in self.episode_records.values():
            for rule in episode.get("rules", []) or []:
                if not isinstance(rule, dict):
                    continue
                text = str(rule.get("text") or "").strip()
                if not text:
                    continue
                metadata = dict(rule.get("metadata") or {})
                flattened.append(
                    {
                        "text": text,
                        "retrieval_text": str(
                            rule.get("retrieval_text")
                            or episode.get("source_question")
                            or ""
                        ),
                        "metadata": metadata,
                    }
                )
        self.feedback_records = flattened

    def add_episode(
        self,
        *,
        episode_id: str,
        source_question: str,
        rules: Iterable[Any],
        metadata: Optional[dict] = None,
        max_rules: Optional[int] = None,
    ) -> int:
        record = self._build_episode_record(
            episode_id=episode_id,
            source_question=source_question,
            rules=rules,
            metadata=metadata,
            max_rules=max_rules,
        )
        if not record["rules"]:
            return 0
        self.episode_records[str(record["episode_id"])] = record
        self._rebuild_feedback_records()
        return len(record["rules"])

    def add(self, memory: str, metadata: Optional[dict] = None) -> None:
        self.add_many([(memory, metadata)])

    def add_many(
        self,
        memories: Iterable[tuple[str, Optional[dict]]],
    ) -> int:
        """Compatibility API that groups rules into question episodes."""
        grouped: Dict[str, Dict[str, Any]] = {}
        for memory, metadata in memories:
            text = str(memory).strip()
            meta = dict(metadata or {})
            if not text or not self._eligible_history_metadata(meta):
                continue
            source_question = str(meta.get("source_question") or "").strip()
            if not source_question:
                continue
            episode_id = self._episode_id(source_question, meta)
            group = grouped.setdefault(
                episode_id,
                {
                    "source_question": source_question,
                    "metadata": dict(meta),
                    "rules": [],
                },
            )
            group["rules"].append({"text": text, "metadata": meta})

        written = 0
        for episode_id, group in grouped.items():
            written += self.add_episode(
                episode_id=episode_id,
                source_question=group["source_question"],
                rules=group["rules"],
                metadata=group["metadata"],
            )
        return written

    def _available_count(self) -> int:
        return len(self.episode_records) + len(self.seed_records)

    @staticmethod
    def _episode_rule_candidates(
        episode: Dict[str, Any],
        *,
        relevance: float,
        min_support: float,
        relevance_method: str,
        ranking_weights: Optional[Dict[str, float]] = None,
        allow_all_history: bool = False,
    ) -> tuple[list[Dict[str, Any]], int]:
        """Return ALL eligible rules for one retrieved episode in quality order.

        Diversity is enforced by ``_adaptive_episode_rule_fill`` rather than by
        hard-truncating every episode here.  This lets ``max_rules_per_episode``
        act as a first-pass diversity quota while still allowing high-quality
        overflow rules to fill an otherwise under-supplied retrieval request.
        """
        rules = []
        rejected_low_support = 0
        for rule in episode.get("rules", []) or []:
            if not isinstance(rule, dict):
                continue
            text = str(rule.get("text") or "").strip()
            metadata = dict(rule.get("metadata") or {})
            if not text:
                continue
            if allow_all_history:
                if not (
                    metadata.get("source_final_safety_valid", False)
                    and metadata.get("source_outcome_verified", False)
                    and str(metadata.get("source_transition") or "").lower() in {
                        "unsafe_to_safe", "safe_to_safe", "unsafe_to_unsafe", "safe_to_unsafe"
                    }
                ):
                    continue
            elif not LongTermMemory._is_successful_correction_metadata(metadata):
                continue
            support = _feedback_support(metadata)
            if support < float(min_support):
                rejected_low_support += 1
                continue
            candidate = _record(
                text,
                metadata,
                relevance,
                relevance_method,
                retrieval_text=str(episode.get("source_question") or ""),
                ranking_weights=ranking_weights,
            )
            candidate["episode_id"] = episode.get("episode_id")
            rules.append(candidate)

        rules.sort(
            key=lambda item: (
                float(item.get("ranking_score", 0.0)),
                float(item.get("feedback_support", 0.0)),
            ),
            reverse=True,
        )
        return rules, rejected_low_support

    @staticmethod
    def _adaptive_episode_rule_fill(
        per_episode_rules: list[tuple[str, list[Dict[str, Any]]]],
        *,
        requested_k: int,
        max_rules_per_episode: int,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, int]], int]:
        """Select rules with diversity first, then fill from episode overflow.

        Pass 1 takes at most ``max_rules_per_episode`` rules from every selected
        episode.  Pass 2 is activated only when that diverse pool is smaller than
        ``requested_k``; it then takes the best remaining rules across those same
        episodes.  Thus the per-episode value is a diversity preference, not a hard
        ceiling that can starve the critic of otherwise valid correction history.
        """
        requested_k = max(0, int(requested_k))
        quota = max(1, int(max_rules_per_episode))
        if requested_k <= 0:
            return [], {}, 0

        primary: list[Dict[str, Any]] = []
        overflow: list[Dict[str, Any]] = []
        counts: Dict[str, Dict[str, int]] = {}

        for episode_id, rules in per_episode_rules:
            clean_rules = [dict(item) for item in rules if isinstance(item, dict)]
            first_pass = clean_rules[:quota]
            remainder = clean_rules[quota:]
            primary.extend(first_pass)
            overflow.extend(remainder)
            counts[str(episode_id)] = {
                "eligible": len(clean_rules),
                "first_pass": len(first_pass),
                "overflow_available": len(remainder),
                "selected": 0,
                "overflow_selected": 0,
            }

        score_key = lambda item: (
            float(item.get("ranking_score", 0.0) or 0.0),
            float(item.get("relevance", 0.0) or 0.0),
            float(item.get("feedback_support", 0.0) or 0.0),
        )
        primary.sort(key=score_key, reverse=True)
        overflow.sort(key=score_key, reverse=True)

        selected = primary[:requested_k]

        # Count by episode_id so the audit remains stable across copied records.
        for item in selected:
            episode_id = str(item.get("episode_id") or "")
            if episode_id in counts:
                counts[episode_id]["selected"] += 1

        overflow_used = 0
        if len(selected) < requested_k:
            needed = requested_k - len(selected)
            extra = overflow[:needed]
            selected.extend(extra)
            overflow_used = len(extra)
            for item in extra:
                episode_id = str(item.get("episode_id") or "")
                if episode_id in counts:
                    counts[episode_id]["selected"] += 1
                    counts[episode_id]["overflow_selected"] += 1

        return selected, counts, overflow_used

    def select_experiences_with_audit(
        self,
        *,
        question: str,
        category: Optional[str] = None,
        k: int = 5,
        min_relevance: float = 0.35,
        min_support: float = 0.50,
        category_scope: str = "prefer",
        require_final_safe: bool = True,
        prefer_unsafe_to_safe: bool = True,
        episode_k: int = 5,
        max_rules_per_episode: int = 2,
        ranking_weights: Optional[Dict[str, float]] = None,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        """Category-based fallback for the two-level episode/rule structure."""
        requested_k = max(0, int(k))
        episode_k = max(1, int(episode_k))
        max_rules_per_episode = max(1, int(max_rules_per_episode))
        ranking_weights = dict(ranking_weights or DEFAULT_RANKING_WEIGHTS)
        category_scope = str(category_scope or "prefer").lower()
        if category_scope not in {"prefer", "same", "other", "any"}:
            raise ValueError(f"Unsupported category_scope: {category_scope!r}")

        episode_hits: list[tuple[Dict[str, Any], float]] = []
        rejected_category = 0
        for episode in self.episode_records.values():
            metadata = dict(episode.get("metadata") or {})
            if not self._eligible_history_metadata(metadata):
                continue
            if require_final_safe and not self._is_successful_correction_metadata(metadata):
                continue
            same_category = _category_matches(metadata, category)
            if category_scope == "same" and not same_category:
                rejected_category += 1
                continue
            if category_scope == "other" and (category is None or same_category):
                rejected_category += 1
                continue

            if category_scope == "same":
                relevance = 1.0
            elif category_scope == "other":
                relevance = 0.60
            elif category is not None and same_category:
                relevance = 1.0
            elif category_scope == "any" or category is None:
                relevance = 0.60
            else:
                relevance = 0.60

            if relevance >= float(min_relevance):
                episode_hits.append((episode, relevance))

        episode_hits.sort(
            key=lambda item: (
                item[1],
                max(
                    (
                        _feedback_support(rule.get("metadata") or {})
                        for rule in (item[0].get("rules") or [])
                        if isinstance(rule, dict)
                    ),
                    default=0.0,
                ),
            ),
            reverse=True,
        )
        selected_episode_hits = episode_hits[:episode_k]

        rejected_low_support = 0
        per_episode_rule_lists: list[tuple[str, list[Dict[str, Any]]]] = []
        for episode, relevance in selected_episode_hits:
            expanded, rejected = self._episode_rule_candidates(
                episode,
                relevance=relevance,
                min_support=min_support,
                relevance_method="episode_category_relevance",
                ranking_weights=ranking_weights,
                allow_all_history=self.allow_all_history,
            )
            rejected_low_support += rejected
            per_episode_rule_lists.append(
                (str(episode.get("episode_id")), expanded)
            )

        candidates, per_episode_fill, overflow_rules_selected = (
            self._adaptive_episode_rule_fill(
                per_episode_rule_lists,
                requested_k=requested_k,
                max_rules_per_episode=max_rules_per_episode,
            )
        )
        per_episode_counts = {
            episode_id: int(detail.get("selected", 0))
            for episode_id, detail in per_episode_fill.items()
        }

        # Question-local seeds remain advisory inputs, not learned historical episodes.
        seed_hits = 0
        for seed_index, seed in enumerate(self.seed_records):
            metadata = dict(seed.get("metadata") or {})
            same_category = _category_matches(metadata, category)
            if category_scope == "same" and not same_category:
                continue
            if category_scope == "other" and (category is None or same_category):
                continue
            relevance = 1.0 if same_category else 0.60
            text = str(seed.get("text") or "").strip()
            if not text or relevance < float(min_relevance):
                continue
            support = _feedback_support(metadata)
            if support < float(min_support):
                continue
            seed_hits += 1
            candidate = _record(
                text,
                metadata,
                relevance,
                "seed_category_relevance",
                retrieval_text=text,
                ranking_weights=ranking_weights,
            )
            candidate["episode_id"] = f"seed:{seed_index}"
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                float(item.get("ranking_score", 0.0)),
                float(item.get("feedback_support", 0.0)),
            ),
            reverse=True,
        )
        selected = candidates[:requested_k]

        audit = {
            "backend": "array_episode_rule_bank",
            "retrieval_attempted": requested_k > 0,
            "retrieval_schema": self.RETRIEVAL_SCHEMA,
            "retrieval_key": "source_question",
            "returned_value": "feedback_rule",
            "query": str(question),
            "category": category,
            "category_scope": category_scope,
            "require_final_safe": bool(require_final_safe),
            "prefer_unsafe_to_safe": bool(prefer_unsafe_to_safe),
            "require_source_transition": (
                "any_valid_committed_transition"
                if self.allow_all_history and not require_final_safe else "unsafe_to_safe"
            ),
            "requested_backend_k": requested_k,
            "requested_episode_k": episode_k,
            "max_rules_per_episode": max_rules_per_episode,
            "ranking_weights": ranking_weights,
            "max_rules_per_episode_semantics": "first_pass_diversity_quota_with_overflow_fill",
            "overflow_rules_selected": overflow_rules_selected,
            "episode_records_available": len(self.episode_records),
            "episode_hits_after_scope": len(episode_hits),
            "episodes_selected": len(selected_episode_hits),
            "selected_episode_ids": [
                str(ep.get("episode_id")) for ep, _ in selected_episode_hits
            ],
            "per_episode_rule_candidates": per_episode_counts,
            "per_episode_rule_fill": per_episode_fill,
            "seed_candidate_hits": seed_hits,
            "rejected_category_scope": rejected_category,
            "rejected_low_support": rejected_low_support,
            "backend_returned_items": len(selected),
            "selected_items": len(selected),
            "min_relevance": float(min_relevance),
            "min_support": float(min_support),
            "selected_relevance": _relevance_summary(
                item.get("relevance", 0.0) for item in selected
            ),
            "zero_result_reason": None if selected else (
                "empty_episode_memory"
                if not self.episode_records and not self.seed_records
                else "no_eligible_history_passed_filters"
            ),
        }
        return selected, audit

    def select_experiences(self, **kwargs) -> list[Dict[str, Any]]:
        selected, _audit = self.select_experiences_with_audit(**kwargs)
        return selected

    def save(self, file_path: str) -> None:
        """Export learned successful-correction rules as flattened JSONL."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self.feedback_records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def load(self, file_path: str) -> None:
        """Load flattened JSONL and rebuild question episodes."""
        path = Path(file_path)
        if not path.exists():
            return

        groups: Dict[str, Dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid long-term memory JSONL at line {line_number}"
                    ) from exc

                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                metadata = dict(item.get("metadata") or {})
                if not text or not self._eligible_history_metadata(metadata):
                    continue
                source_question = str(
                    item.get("retrieval_text")
                    or metadata.get("source_question")
                    or ""
                ).strip()
                if not source_question:
                    continue
                episode_id = self._episode_id(source_question, metadata)
                group = groups.setdefault(
                    episode_id,
                    {
                        "source_question": source_question,
                        "metadata": dict(metadata),
                        "rules": [],
                    },
                )
                group["rules"].append({"text": text, "metadata": metadata})

        self.episode_records = {}
        for episode_id, group in groups.items():
            record = self._build_episode_record(
                episode_id=episode_id,
                source_question=group["source_question"],
                rules=group["rules"],
                metadata=group["metadata"],
            )
            if record["rules"]:
                self.episode_records[episode_id] = record
        self._rebuild_feedback_records()

    def __len__(self) -> int:
        return len(self.feedback_records)

    def __str__(self) -> str:
        if not self.feedback_records and not self.seed_records:
            return f"{self.description}:\n  No memories stored."
        records = [*self.seed_records, *self.feedback_records]
        return self.description + ":\n" + "\n".join(
            f"  {item['text']}" for item in records
        )


class VectorStoreMemory(LongTermMemory):
    """Two-level Chroma episode index + local persistent rule bank.

    Chroma stores exactly one vector document per successful historical question.
    The rule bank stores the reusable critique rules attached to that episode.

        Chroma page_content = historical source question
        Chroma metadata     = episode-level provenance
        rule bank           = episode_id -> successful UNSAFE->SAFE critique rules
    """

    def __init__(
        self,
        description: str = "Long-Term Memory",
        index_name: str = "red-debate-memory",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        allow_all_history: bool = False,
    ) -> None:
        super().__init__(description, allow_all_history=allow_all_history)
        self.persist_directory = (
            persist_directory
            or os.environ.get("CHROMA_DB_ROOT")
            or "./chroma_db"
        )
        os.makedirs(self.persist_directory, exist_ok=True)
        self.index_name = str(index_name)

        try:
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "VectorStoreMemory requires langchain-chroma and langchain-openai"
            ) from exc

        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Chroma(
            collection_name=self.index_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        try:
            collection_metadata = self.vector_store._collection.metadata or {}
        except Exception:
            collection_metadata = {}
        self.distance_metric = str(
            collection_metadata.get("hnsw:space", "l2")
        ).lower()
        if self.distance_metric not in {"cosine", "l2", "ip"}:
            self.distance_metric = "l2"

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.index_name)
        self.rule_bank_path = (
            Path(self.persist_directory)
            / f"{safe_name}.rule_bank.jsonl"
        )
        self._load_rule_bank_journal()

    def embed_query(self, text: str) -> list[float]:
        return list(self.embeddings.embed_query(str(text)))

    def _distance_to_relevance(self, raw_distance: float) -> float:
        distance = float(raw_distance)
        if self.distance_metric == "l2":
            relevance = 1.0 - (distance / 2.0)
        else:
            relevance = 1.0 - distance
        return max(0.0, min(1.0, relevance))

    @staticmethod
    def _clean_metadata(metadata: Optional[dict]) -> dict:
        result: Dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                result[str(key)] = value
        return result

    def _load_rule_bank_journal(self) -> None:
        if not self.rule_bank_path.exists():
            return
        latest: Dict[str, Dict[str, Any]] = {}
        with self.rule_bank_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid rule-bank JSONL at line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    continue
                episode_id = str(record.get("episode_id") or "").strip()
                if not episode_id:
                    continue
                latest[episode_id] = record
        self.episode_records = latest
        self._rebuild_feedback_records()

    def _append_rule_bank_journal(self, record: Dict[str, Any]) -> None:
        self.rule_bank_path.parent.mkdir(parents=True, exist_ok=True)
        with self.rule_bank_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _vector_document_id(episode_id: str) -> str:
        return hashlib.sha256(
            f"unsafe_to_safe_episode|{episode_id}".encode("utf-8")
        ).hexdigest()

    def add_episode(
        self,
        *,
        episode_id: str,
        source_question: str,
        rules: Iterable[Any],
        metadata: Optional[dict] = None,
        max_rules: Optional[int] = None,
    ) -> int:
        record = self._build_episode_record(
            episode_id=episode_id,
            source_question=source_question,
            rules=rules,
            metadata=metadata,
            max_rules=max_rules,
        )
        if not record["rules"]:
            return 0

        episode_metadata = dict(record["metadata"])
        episode_metadata["rule_count"] = len(record["rules"])
        chroma_metadata = self._clean_metadata(episode_metadata)
        retrieval_text = str(record["source_question"])
        vector = self.embeddings.embed_documents([retrieval_text])[0]
        document_id = self._vector_document_id(str(record["episode_id"]))

        collection = getattr(self.vector_store, "_collection", None)
        if collection is not None and hasattr(collection, "upsert"):
            collection.upsert(
                ids=[document_id],
                documents=[retrieval_text],
                metadatas=[chroma_metadata],
                embeddings=[vector],
            )
        else:
            from langchain_core.documents import Document

            self.vector_store.add_documents(
                [
                    Document(
                        page_content=retrieval_text,
                        metadata=chroma_metadata,
                    )
                ],
                ids=[document_id],
            )

        # Update memory only after the vector upsert succeeds.
        self.episode_records[str(record["episode_id"])] = record
        self._rebuild_feedback_records()
        self._append_rule_bank_journal(record)
        return len(record["rules"])

    def _available_count(self) -> int:
        try:
            return int(self.vector_store._collection.count()) + len(self.seed_records)
        except Exception:
            return len(self.episode_records) + len(self.seed_records)

    def _search_with_error(
        self,
        question: str,
        k: int,
        filter_: Optional[dict] = None,
    ) -> tuple[list[tuple[str, dict, float]], Optional[str]]:
        kwargs: Dict[str, Any] = {"k": max(1, int(k))}
        if filter_ is not None:
            kwargs["filter"] = filter_
        try:
            results = self.vector_store.similarity_search_with_score(
                str(question), **kwargs
            )
        except Exception as exc:
            return [], repr(exc)

        output: list[tuple[str, dict, float]] = []
        for document, distance in results:
            try:
                relevance = self._distance_to_relevance(float(distance))
            except (TypeError, ValueError):
                continue
            output.append(
                (
                    str(document.page_content),
                    dict(document.metadata or {}),
                    relevance,
                )
            )
        return output, None

    def select_experiences_with_audit(
        self,
        *,
        question: str,
        category: Optional[str] = None,
        k: int = 5,
        min_relevance: float = 0.35,
        min_support: float = 0.50,
        category_scope: str = "prefer",
        require_final_safe: bool = True,
        prefer_unsafe_to_safe: bool = True,
        episode_k: int = 5,
        max_rules_per_episode: int = 2,
        ranking_weights: Optional[Dict[str, float]] = None,
    ) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        requested_k = max(0, int(k))
        episode_k = max(1, int(episode_k))
        max_rules_per_episode = max(1, int(max_rules_per_episode))
        ranking_weights = dict(ranking_weights or DEFAULT_RANKING_WEIGHTS)
        category_scope = str(category_scope or "prefer").lower()
        if category_scope not in {"prefer", "same", "other", "any"}:
            raise ValueError(f"Unsupported category_scope: {category_scope!r}")

        # Retrieve episodes, not rules.  Query a larger pool because category and
        # quality gates can remove some hits before the episode_k cap is filled.
        pool_size = max(10, episode_k * 4)

        clauses: list[dict] = (
            [] if self.allow_all_history and not require_final_safe else [
                {"source_transition": "unsafe_to_safe"},
                {"source_final_safety": "safe"},
            ]
        )
        if category is not None and category_scope == "same":
            clauses.append({"category": str(category)})
        elif category is not None and category_scope == "other":
            clauses.append({"category": {"$ne": str(category)}})

        filter_: Optional[dict]
        if not clauses:
            filter_ = None
        elif len(clauses) == 1:
            filter_ = clauses[0]
        else:
            filter_ = {"$and": clauses}

        vector_hits, vector_error = self._search_with_error(
            question,
            pool_size,
            filter_=filter_,
        )
        fallback_filter_used = False
        if vector_error is not None:
            vector_hits, fallback_error = self._search_with_error(
                question,
                pool_size,
                filter_=None,
            )
            fallback_filter_used = True
            vector_error = fallback_error

        dedup_hits: Dict[str, Dict[str, Any]] = {}
        rejected_missing_rule_bank = 0
        rejected_low_relevance = 0
        rejected_category = 0
        rejected_non_correction_episode = 0

        for retrieval_text, metadata, relevance in vector_hits:
            episode_id = str(metadata.get("episode_id") or "").strip()
            if not episode_id:
                continue
            episode = self.episode_records.get(episode_id)
            if not isinstance(episode, dict):
                rejected_missing_rule_bank += 1
                continue

            episode_metadata = dict(episode.get("metadata") or {})
            if not self._eligible_history_metadata(episode_metadata):
                rejected_non_correction_episode += 1
                continue
            if require_final_safe and not self._is_successful_correction_metadata(episode_metadata):
                rejected_non_correction_episode += 1
                continue

            same_category = _category_matches(episode_metadata, category)
            if category_scope == "same" and not same_category:
                rejected_category += 1
                continue
            if category_scope == "other" and (category is None or same_category):
                rejected_category += 1
                continue
            if relevance < float(min_relevance):
                rejected_low_relevance += 1
                continue

            previous = dedup_hits.get(episode_id)
            if previous is None or relevance > float(previous["relevance"]):
                dedup_hits[episode_id] = {
                    "episode": episode,
                    "retrieval_text": retrieval_text,
                    "relevance": relevance,
                }

        ordered_episode_hits = sorted(
            dedup_hits.values(),
            key=lambda item: float(item["relevance"]),
            reverse=True,
        )
        selected_episode_hits = ordered_episode_hits[:episode_k]

        rejected_low_support = 0
        per_episode_rule_lists: list[tuple[str, list[Dict[str, Any]]]] = []
        for hit in selected_episode_hits:
            episode = hit["episode"]
            expanded, rejected = self._episode_rule_candidates(
                episode,
                relevance=float(hit["relevance"]),
                min_support=min_support,
                relevance_method="source_question_episode_vector",
                ranking_weights=ranking_weights,
                allow_all_history=self.allow_all_history,
            )
            rejected_low_support += rejected
            per_episode_rule_lists.append(
                (str(episode.get("episode_id")), expanded)
            )

        candidates, per_episode_fill, overflow_rules_selected = (
            self._adaptive_episode_rule_fill(
                per_episode_rule_lists,
                requested_k=requested_k,
                max_rules_per_episode=max_rules_per_episode,
            )
        )
        per_episode_counts = {
            episode_id: int(detail.get("selected", 0))
            for episode_id, detail in per_episode_fill.items()
        }

        # Preserve current-question seed memories as advisory non-historical rules.
        seed_hits = 0
        for seed_index, seed in enumerate(self.seed_records):
            metadata = dict(seed.get("metadata") or {})
            same_category = _category_matches(metadata, category)
            if category_scope == "same" and not same_category:
                continue
            if category_scope == "other" and (category is None or same_category):
                continue
            relevance = 1.0 if same_category else 0.60
            if relevance < float(min_relevance):
                continue
            text = str(seed.get("text") or "").strip()
            if not text or _feedback_support(metadata) < float(min_support):
                continue
            candidate = _record(
                text,
                metadata,
                relevance,
                "seed_category_relevance",
                retrieval_text=text,
                ranking_weights=ranking_weights,
            )
            candidate["episode_id"] = f"seed:{seed_index}"
            candidates.append(candidate)
            seed_hits += 1

        candidates.sort(
            key=lambda item: (
                float(item.get("ranking_score", 0.0)),
                float(item.get("feedback_support", 0.0)),
            ),
            reverse=True,
        )
        selected = candidates[:requested_k]

        audit = {
            "backend": "chroma_episode_rule_bank",
            "retrieval_attempted": requested_k > 0,
            "retrieval_schema": self.RETRIEVAL_SCHEMA,
            "retrieval_key": "source_question",
            "returned_value": "feedback_rule",
            "query": str(question),
            "category": category,
            "category_scope": category_scope,
            "require_final_safe": bool(require_final_safe),
            "prefer_unsafe_to_safe": bool(prefer_unsafe_to_safe),
            "require_source_transition": (
                "any_valid_committed_transition"
                if self.allow_all_history and not require_final_safe else "unsafe_to_safe"
            ),
            "requested_backend_k": requested_k,
            "requested_episode_k": episode_k,
            "max_rules_per_episode": max_rules_per_episode,
            "ranking_weights": ranking_weights,
            "max_rules_per_episode_semantics": "first_pass_diversity_quota_with_overflow_fill",
            "overflow_rules_selected": overflow_rules_selected,
            "candidate_pool_size": pool_size,
            "filter": filter_,
            "fallback_filter_used": fallback_filter_used,
            "vector_error": vector_error,
            "episode_records_available": len(self.episode_records),
            "vector_episode_hits": len(vector_hits),
            "unique_eligible_episode_hits": len(ordered_episode_hits),
            "episodes_selected": len(selected_episode_hits),
            "selected_episode_ids": [
                str(hit["episode"].get("episode_id"))
                for hit in selected_episode_hits
            ],
            "per_episode_rule_candidates": per_episode_counts,
            "per_episode_rule_fill": per_episode_fill,
            "seed_candidate_hits": seed_hits,
            "rejected_missing_rule_bank": rejected_missing_rule_bank,
            "rejected_low_relevance": rejected_low_relevance,
            "rejected_low_support": rejected_low_support,
            "rejected_category_scope": rejected_category,
            "rejected_non_unsafe_to_safe_episode": rejected_non_correction_episode,
            "backend_returned_items": len(selected),
            "selected_items": len(selected),
            "min_relevance": float(min_relevance),
            "min_support": float(min_support),
            "selected_relevance": _relevance_summary(
                item.get("relevance", 0.0) for item in selected
            ),
            "zero_result_reason": None if selected else (
                "empty_episode_memory"
                if not self.episode_records and not self.seed_records
                else (
                    "vector_search_error"
                    if vector_error
                    else "no_eligible_history_passed_filters"
                )
            ),
        }
        return selected, audit

    def save(self, file_path: str) -> None:
        # The rule-bank journal and Chroma are already persistent.  The runner's
        # results export remains a flattened rule JSONL for easy analysis.
        super().save(file_path)

    def load(self, file_path: str) -> None:
        # Resume for the vector backend is handled by persistent Chroma + rule-bank
        # journal.  Importing the flattened results JSONL would duplicate episodes.
        return
