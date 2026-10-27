"""Three-round CaRDebate with embedded critique and structured episode state.

The three debate rounds are ANSWER rounds:
- Round 1: independent answer -> evaluator -> critique cycle 1.
- Round 2: revision -> evaluator -> monotonic commit/rollback -> critique cycle 2.
- Round 3: final revision -> evaluator -> monotonic commit/rollback -> feedback.

Critique cycles are internal post-evaluation phases, not standalone debate
rounds. Short-term memory is represented by EpisodeState; models never receive
that state object wholesale and the orchestrator exposes only stage-appropriate
fields. Long-term feedback is generated only for evaluator-verified successful
UNSAFE -> SAFE adjacent committed transitions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import DebateAgent, EvalAgent, FeedbackAgent
from .confidence import (
    enforce_critique_semantics,
    enforce_feedback_transition_semantics,
    estimate_agent_reliability,
    feedback_error_evidence,
    freeze_agent_stats,
    is_safe_evaluation,
    is_valid_evaluation,
    parse_critique_output,
    parse_debater_output,
    parse_evaluator_output,
    parse_feedback_output,
    record_r1_observations,
    text_similarity,
)
from .memory import LongTermMemory
from .utils import setup_logger


PROTOCOL_VERSION = "evaluator_safe_peer_bayesian_question_v7_stm_success_category_memory"


ResponseMap = Dict[str, Dict[str, Any]]


class AnswerGenerationExhausted(RuntimeError):
    """Raised when a critical structured stage exhausts its format retries.

    The historical class name is retained for caller compatibility. Answer rounds,
    critique phases, and evaluator format/parse exhaustion abort only the current question when
    the runner is configured with ``abort_question_on_exhaustion=true``.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: Optional[str] = None,
        agent: Optional[str] = None,
        failure_reason: Optional[str] = None,
        audit: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.agent = agent
        self.failure_reason = failure_reason
        self.audit = dict(audit or {})


@dataclass
class EpisodeState:
    """Structured short-term state for exactly one three-round question.

    The three debate rounds are answer rounds:
    - ``initial``: Round 1 independent answer.
    - ``revision``: Round 2 revised answer.
    - ``final``: Round 3 final answer.

    ``critiques`` is the post-evaluation critique attached to Round 1 and
    ``critique2`` is the post-evaluation critique attached to Round 2. Critiques
    are internal deliberation phases, not standalone debate rounds.

    Historical reliability is frozen before Round 1. The same frozen snapshot is
    used by both critique phases, so the current question cannot update and then
    consume its own Bayesian history.
    """

    initial: ResponseMap = field(default_factory=dict)
    critiques: ResponseMap = field(default_factory=dict)
    revision: ResponseMap = field(default_factory=dict)
    critique2: ResponseMap = field(default_factory=dict)
    final: ResponseMap = field(default_factory=dict)

    # Per-critique-cycle audit namespaces. Keys are ``cycle1`` and ``cycle2``.
    peer_routing: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    history_assistance: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    experience_selection: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)
    memory_retrieval_audit: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    # Current-question only: successful R1-UNSAFE -> R2-SAFE repair provenance.
    repair_memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    reliability_snapshot: Dict[str, Any] = field(default_factory=dict)
    question_embedding: Optional[List[float]] = None

    @property
    def rounds(self) -> List[ResponseMap]:
        """Return only the three answer rounds, never critique phases."""
        result: List[ResponseMap] = []
        if self.initial:
            result.append(self.initial)
        if self.revision:
            result.append(self.revision)
        if self.final:
            result.append(self.final)
        return result

    @property
    def critique_rounds(self) -> List[ResponseMap]:
        result: List[ResponseMap] = []
        if self.critiques:
            result.append(self.critiques)
        if self.critique2:
            result.append(self.critique2)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial": self.initial,
            "critiques": self.critiques,
            "revision": self.revision,
            "critique2": self.critique2,
            "final": self.final,
            "peer_routing": self.peer_routing,
            "history_assistance": self.history_assistance,
            "experience_selection": self.experience_selection,
            "memory_retrieval_audit": self.memory_retrieval_audit,
            "repair_memory": self.repair_memory,
            "reliability_snapshot": self.reliability_snapshot,
            "question_embedding": self.question_embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeState":
        if not isinstance(data, dict):
            raise ValueError("episode_state must be a JSON object")
        return cls(
            initial=dict(data.get("initial") or {}),
            critiques=dict(data.get("critiques") or {}),
            revision=dict(data.get("revision") or {}),
            critique2=dict(data.get("critique2") or {}),
            final=dict(data.get("final") or {}),
            peer_routing=dict(data.get("peer_routing") or {}),
            history_assistance=dict(data.get("history_assistance") or {}),
            experience_selection=dict(data.get("experience_selection") or {}),
            memory_retrieval_audit=dict(data.get("memory_retrieval_audit") or {}),
            repair_memory=dict(data.get("repair_memory") or {}),
            reliability_snapshot=dict(data.get("reliability_snapshot") or {}),
            question_embedding=(
                list(data["question_embedding"])
                if isinstance(data.get("question_embedding"), list)
                else None
            ),
        )


class Debate:
    """Three answer rounds with evaluation and conditional critique embedded.

    Round 1: independent answer -> evaluator -> conditional critique.
    Round 2: revise own Round-1 answer -> evaluator -> conditional critique.
    Round 3: revise own Round-2 answer -> final evaluator -> feedback learning.

    Critique is not a standalone debate round. Both critique phases use the same
    frozen historical Bayesian snapshot. Current evaluator verdicts alone determine
    safe-peer eligibility in each round; the peer graph is rebuilt after Round 2.
    """

    def __init__(
        self,
        question: dict,
        debate_agents: List[DebateAgent],
        eval_agent: EvalAgent,
        feedback_agent: Optional[FeedbackAgent],
        long_term_memory: LongTermMemory,
        *,
        log_file: Optional[str] = None,
        model_output_dir: Optional[str] = None,
        model_output_files: Optional[Dict[str, str]] = None,
        dataset_name: Optional[str] = None,
        question_index: Optional[int] = None,
        agent_stats: Optional[dict] = None,
        memory_top_k: int,
        memory_episode_top_k: int,
        memory_max_rules_per_episode: int,
        memory_max_rules_stored_per_episode: int,
        memory_min_relevance: float,
        memory_min_support: float,
        memory_semantic_dedup_enabled: bool,
        memory_dedup_threshold: float,
        memory_min_marginal_utility: float,
        memory_candidate_oversample_factor: int,
        memory_relevance_weight: float,
        memory_support_weight: float,
        memory_outcome_priority_weight: float,
        answer_max_words: int,
        safe_peer_history_min_items: int,
        safe_peer_history_max_items: int,
        no_safe_peer_history_min_items: int,
        no_safe_peer_history_max_items: int,
        cross_category_history_min_items: int,
        cross_category_history_max_items: int,
        successful_history_only: bool,
        prefer_unsafe_to_safe_history: bool,
        enable_verified_repair_memory: bool,
        format_retry_max_retries: int,
        format_retry_reasons: List[str],
        reliability_prior_alpha: float,
        reliability_prior_beta: float,
        reliability_credible_level: float,
        reliability_similar_history_k: int,
        reliability_min_similarity: float,
        use_expected_safety_mass: bool,
        unsafe_sensitivity: float,
        safe_specificity: float,
        evaluator_prior_floor: float,
        evaluator_prior_ceiling: float,
        max_safe_peers: int,
        feedback_min_quality: float,
        feedback_min_confidence: float,
        feedback_support_quality_weight: float,
        feedback_support_confidence_weight: float,
        monotonic_commit_enabled: bool,
        monotonic_safety_score_tolerance: float,
        monotonic_safety_coverage_tolerance: float,
        history_enabled: bool = True,
        feedback_enabled: bool = True,
        history_mode: str = "cardebate",
    ) -> None:
        if eval_agent is None:
            raise ValueError(f"{PROTOCOL_VERSION} requires an evaluator")
        if not debate_agents:
            raise ValueError("At least one debate agent is required")

        self.question = question
        self.question_text = (
            str(question.get("text", ""))
            if isinstance(question, dict)
            else str(question)
        ).strip()
        if not self.question_text:
            raise ValueError("Question text must not be empty")
        self.question_label = (
            question.get("label") if isinstance(question, dict) else None
        )

        self.debate_agents = list(debate_agents)
        self.eval_agent = eval_agent
        self.feedback_agent = feedback_agent
        self.long_term_memory = long_term_memory
        self.agent_stats = agent_stats if isinstance(agent_stats, dict) else {}
        self.answer_max_words = int(answer_max_words)
        if self.answer_max_words <= 0:
            raise ValueError("answer_max_words must be > 0")

        self.state = EpisodeState()
        self.feedback_raw: str = ""
        self.feedback_details: Dict[str, Any] = {
            "status": "not_run",
            "stored_rules": [],
        }

        # Memory/retrieval hyperparameters are required from YAML via the launcher.
        self.memory_top_k = int(memory_top_k)
        self.memory_episode_top_k = int(memory_episode_top_k)
        self.memory_max_rules_per_episode = int(memory_max_rules_per_episode)
        self.memory_max_rules_stored_per_episode = int(
            memory_max_rules_stored_per_episode
        )
        self.memory_min_relevance = float(memory_min_relevance)
        self.memory_min_support = float(memory_min_support)
        self.memory_semantic_dedup_enabled = bool(memory_semantic_dedup_enabled)
        self.memory_dedup_threshold = float(memory_dedup_threshold)
        self.memory_min_marginal_utility = float(memory_min_marginal_utility)
        self.memory_candidate_oversample_factor = int(
            memory_candidate_oversample_factor
        )
        self.memory_relevance_weight = float(memory_relevance_weight)
        self.memory_support_weight = float(memory_support_weight)
        self.memory_outcome_priority_weight = float(memory_outcome_priority_weight)

        if not 1 <= self.memory_top_k <= 5:
            raise ValueError("memory_top_k must be between 1 and 5")
        if not 1 <= self.memory_episode_top_k <= 20:
            raise ValueError("memory_episode_top_k must be between 1 and 20")
        if not 1 <= self.memory_max_rules_per_episode <= 5:
            raise ValueError("memory_max_rules_per_episode must be between 1 and 5")
        if not 1 <= self.memory_max_rules_stored_per_episode <= 20:
            raise ValueError(
                "memory_max_rules_stored_per_episode must be between 1 and 20"
            )
        if self.memory_candidate_oversample_factor < 1:
            raise ValueError("memory_candidate_oversample_factor must be >= 1")
        if abs(
            self.memory_relevance_weight
            + self.memory_support_weight
            + self.memory_outcome_priority_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "memory ranking weights must sum to 1.0"
            )

        self.safe_peer_history_min_items = int(safe_peer_history_min_items)
        self.safe_peer_history_max_items = int(safe_peer_history_max_items)
        self.no_safe_peer_history_min_items = int(no_safe_peer_history_min_items)
        self.no_safe_peer_history_max_items = int(no_safe_peer_history_max_items)
        self.cross_category_history_min_items = int(cross_category_history_min_items)
        self.cross_category_history_max_items = int(cross_category_history_max_items)
        self.successful_history_only = bool(successful_history_only)
        self.history_enabled = bool(history_enabled)
        self.feedback_enabled = bool(feedback_enabled)
        self.history_mode = str(history_mode)
        self.prefer_unsafe_to_safe_history = bool(prefer_unsafe_to_safe_history)
        self.enable_verified_repair_memory = bool(enable_verified_repair_memory)

        history_ranges = {
            "safe_peer": (
                self.safe_peer_history_min_items,
                self.safe_peer_history_max_items,
            ),
            "no_safe_peer": (
                self.no_safe_peer_history_min_items,
                self.no_safe_peer_history_max_items,
            ),
            "cross_category": (
                self.cross_category_history_min_items,
                self.cross_category_history_max_items,
            ),
        }
        for branch, (min_items, max_items) in history_ranges.items():
            if min_items < 1 or max_items < min_items:
                raise ValueError(
                    f"{branch} history range must satisfy 1 <= min_items <= max_items"
                )
            if max_items > self.memory_top_k:
                raise ValueError(
                    f"{branch} history max_items ({max_items}) must not exceed "
                    f"memory_top_k ({self.memory_top_k})"
                )

        self.format_retry_max_retries = int(format_retry_max_retries)
        if self.format_retry_max_retries < 0:
            raise ValueError("format_retry_max_retries must be >= 0")
        self.answer_max_attempts = 1 + self.format_retry_max_retries
        self.format_retry_reasons = {
            str(item).strip() for item in format_retry_reasons if str(item).strip()
        }
        if not self.format_retry_reasons:
            raise ValueError("format_retry_reasons must be a non-empty list")

        self.reliability_prior_alpha = float(reliability_prior_alpha)
        self.reliability_prior_beta = float(reliability_prior_beta)
        self.reliability_credible_level = float(reliability_credible_level)
        self.reliability_similar_history_k = int(reliability_similar_history_k)
        self.reliability_min_similarity = float(reliability_min_similarity)
        if self.reliability_similar_history_k < 0:
            raise ValueError("reliability_similar_history_k must be >= 0")

        self.use_expected_safety_mass = bool(use_expected_safety_mass)
        self.unsafe_sensitivity = float(unsafe_sensitivity)
        self.safe_specificity = float(safe_specificity)
        self.evaluator_prior_floor = float(evaluator_prior_floor)
        self.evaluator_prior_ceiling = float(evaluator_prior_ceiling)

        self.max_safe_peers = int(max_safe_peers)
        if self.max_safe_peers < 0:
            raise ValueError("max_safe_peers must be >= 0")

        self.feedback_min_quality = float(feedback_min_quality)
        self.feedback_min_confidence = float(feedback_min_confidence)
        self.feedback_support_quality_weight = float(
            feedback_support_quality_weight
        )
        self.feedback_support_confidence_weight = float(
            feedback_support_confidence_weight
        )
        if abs(
            self.feedback_support_quality_weight
            + self.feedback_support_confidence_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "feedback_support_quality_weight + "
                "feedback_support_confidence_weight must equal 1.0"
            )

        self.monotonic_commit_enabled = bool(monotonic_commit_enabled)

        self.monotonic_safety_score_tolerance = float(
            monotonic_safety_score_tolerance
        )
        self.monotonic_safety_coverage_tolerance = float(
            monotonic_safety_coverage_tolerance
        )

        probability_fields = {
            "memory_min_relevance": self.memory_min_relevance,
            "memory_min_support": self.memory_min_support,
            "memory_dedup_threshold": self.memory_dedup_threshold,
            "memory_min_marginal_utility": self.memory_min_marginal_utility,
            "memory_relevance_weight": self.memory_relevance_weight,
            "memory_support_weight": self.memory_support_weight,
            "memory_outcome_priority_weight": self.memory_outcome_priority_weight,
            "reliability_min_similarity": self.reliability_min_similarity,
            "unsafe_sensitivity": self.unsafe_sensitivity,
            "safe_specificity": self.safe_specificity,
            "evaluator_prior_floor": self.evaluator_prior_floor,
            "evaluator_prior_ceiling": self.evaluator_prior_ceiling,
            "feedback_min_quality": self.feedback_min_quality,
            "feedback_min_confidence": self.feedback_min_confidence,
            "feedback_support_quality_weight": self.feedback_support_quality_weight,
            "feedback_support_confidence_weight": self.feedback_support_confidence_weight,
            "monotonic_safety_score_tolerance": self.monotonic_safety_score_tolerance,
            "monotonic_safety_coverage_tolerance": self.monotonic_safety_coverage_tolerance,
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.evaluator_prior_floor > self.evaluator_prior_ceiling:
            raise ValueError("evaluator_prior_floor must not exceed ceiling")
        if self.reliability_prior_alpha <= 0.0:
            raise ValueError("reliability_prior_alpha must be positive")
        if self.reliability_prior_beta <= 0.0:
            raise ValueError("reliability_prior_beta must be positive")
        if not 0.5 <= self.reliability_credible_level < 1.0:
            raise ValueError("reliability_credible_level must be in [0.5, 1.0)")

        self.protocol_version = PROTOCOL_VERSION
        self.log_file = log_file
        self.logger = setup_logger(log_file) if log_file else logging.getLogger(__name__)
        self.model_output_dir = (
            Path(model_output_dir).expanduser().resolve()
            if model_output_dir
            else None
        )
        if self.model_output_dir is not None:
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_files = {
            str(key): str(value)
            for key, value in dict(model_output_files or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self.dataset_name = str(dataset_name) if dataset_name is not None else None
        self.question_index = (
            int(question_index) if question_index is not None else None
        )
        self._main_agent_names = {agent.name for agent in self.debate_agents}

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_question(self):
        return self.question

    def get_question_text(self) -> str:
        return self.question_text

    def get_question_label(self):
        return self.question_label

    def get_responses(self):
        return self.state.rounds

    def get_episode_state(self) -> EpisodeState:
        return self.state

    def get_feedback(self) -> Dict[str, Any]:
        return self.feedback_details

    def get_feedback_raw(self) -> str:
        return self.feedback_raw

    def get_stored_feedback_rules(self) -> list[str]:
        # LTM stores outcome provenance per successful source trajectory, so the same
        # rule text may be persisted more than once with different source-agent metadata.
        # PEFT consumers receive each textual rule only once.
        result: list[str] = []
        seen = set()
        for item in self.feedback_details.get("stored_rules", []) or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("rule", "")).strip()
            key = " ".join(text.lower().split())
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result

    def _question_identifier(self) -> Optional[str]:
        if not isinstance(self.question, dict):
            return None
        for key in ("id", "question_id", "idx", "index"):
            value = self.question.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None

    def _memory_episode_id(self) -> str:
        if self.dataset_name is not None and self.question_index is not None:
            return f"{self.dataset_name}:{self.question_index}"
        explicit = self._question_identifier()
        if explicit:
            return str(explicit)
        return "episode:" + hashlib.sha256(
            self.question_text.encode("utf-8")
        ).hexdigest()[:24]

    def _write_agent_jsonl(
        self,
        agent_name: str,
        record: Dict[str, Any],
    ) -> None:
        """Append one structured audit event to the configured model-output file.

        JSONL is written by the orchestrator rather than by ``agents.py`` so the
        record can contain the actual policy state that controlled each call:
        evaluator-only peer routing, Bayesian history access, selected long-term memories,
        parser results, retries, and final evaluator decisions.
        """
        if self.model_output_dir is None:
            return

        safe_name = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in str(agent_name)
        )
        filename = self.model_output_files.get(
            str(agent_name), f"{safe_name}.jsonl"
        )
        filename_path = Path(str(filename))
        if filename_path.is_absolute() or filename_path.name != str(filename):
            self.logger.error(
                "Invalid model output filename for %s: %r",
                agent_name,
                filename,
            )
            return
        path = self.model_output_dir / filename_path.name
        now = time.time()
        payload = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime(now)
            ),
            "timestamp_unix": now,
            "protocol_version": self.protocol_version,
            "dataset": self.dataset_name,
            "question_index": self.question_index,
            "question_id": self._question_identifier(),
            "category": self.question_label,
            "question": self.question_text,
            "agent": str(agent_name),
            **dict(record),
        }

        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        payload, ensure_ascii=False, default=str
                    )
                    + "\n"
                )
        except Exception as exc:
            self.logger.error(
                "Failed to append per-agent JSONL %s: %s", path, exc
            )

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    def start(self, rounds: int = 3) -> None:
        """Run exactly three answer rounds with barrier-synchronized evaluation.

        All three agents finish an answer round before any critique routing for
        that round is constructed. This prevents execution order from changing
        the evaluator-safe peer set.
        """
        if int(rounds) != 3:
            raise ValueError(f"{self.protocol_version} requires exactly 3 rounds")

        # Freeze historical state exactly once for the entire question.
        self.state.reliability_snapshot = freeze_agent_stats(self.agent_stats)
        self.state.question_embedding = self._embed_question(self.question_text)

        # Round 1 = independent answer -> evaluate -> critique.
        self.logger.info("Round 1: independent answers -> evaluation -> critique")
        self.state.initial = self._run_initial_round()
        self._evaluate_round(self.state.initial, stage="initial")
        self.state.critiques = self._run_critique_round(
            self.state.initial,
            critique_cycle=1,
            source_round=1,
        )

        # Round 2 = revise -> evaluate -> critique. Routing is rebuilt from Round-2
        # evaluator verdicts; no Round-1 peer graph is reused.
        self.logger.info("Round 2: revision -> evaluation -> critique")
        self.state.revision = self._run_revision_round(
            self.state.initial,
            self.state.critiques,
            stage="revision",
            round_index=2,
        )
        self._evaluate_round(
            self.state.revision,
            stage="revision",
            reuse_from=self.state.initial,
        )
        self._apply_monotonic_safety_commit(
            self.state.initial,
            self.state.revision,
            stage="revision",
        )
        self._update_verified_repair_memory()
        self.state.critique2 = self._run_critique_round(
            self.state.revision,
            critique_cycle=2,
            source_round=2,
        )

        # Round 3 = final revision -> final evaluation -> feedback.
        self.logger.info("Round 3: final revision -> evaluation -> feedback")
        self.state.final = self._run_revision_round(
            self.state.revision,
            self.state.critique2,
            stage="final",
            round_index=3,
        )
        self._evaluate_round(
            self.state.final,
            stage="final",
            reuse_from=self.state.revision,
        )
        self._apply_monotonic_safety_commit(
            self.state.revision,
            self.state.final,
            stage="final",
        )

        self._generate_targeted_feedback(
            stages={"initial", "revision", "final"}
        )

        # Bayesian reliability remains explicitly R1 reliability. Only Round-1
        # evaluator observations are written after the whole question completes.
        record_r1_observations(
            self.agent_stats,
            self.question_text,
            self.state.initial,
            question_embedding=self.state.question_embedding,
            use_expected_safety_mass=self.use_expected_safety_mass,
            unsafe_sensitivity=self.unsafe_sensitivity,
            safe_specificity=self.safe_specificity,
            prior_floor=self.evaluator_prior_floor,
            prior_ceiling=self.evaluator_prior_ceiling,
        )

        self.logger.info("Debate completed")

    def _update_verified_repair_memory(self) -> None:
        """Record successful R1-UNSAFE -> R2-SAFE repairs as short-term memory.

        This state never enters Chroma and is never available to future questions.
        It exists only so Cycle-2 preservation and Round-3 revision can retain the
        constraints that just produced a verified safe answer on this same question.
        """
        if not self.enable_verified_repair_memory:
            self.state.repair_memory = {}
            return
        repair_memory: Dict[str, Dict[str, Any]] = {}
        for agent in self.debate_agents:
            name = agent.name
            initial = self.state.initial.get(name, {})
            revision = self.state.revision.get(name, {})
            initial_eval = initial.get("evaluation")
            revision_eval = revision.get("evaluation")
            if not (is_valid_evaluation(initial_eval) and is_valid_evaluation(revision_eval)):
                continue
            if initial_eval.get("safety") != "unsafe" or revision_eval.get("safety") != "safe":
                continue
            critique_record = self.state.critiques.get(name, {})
            critique = critique_record.get("critique") or {}
            rules = [str(x).strip() for x in (critique.get("safety_rules") or []) if str(x).strip()]
            findings = [dict(x) for x in (critique.get("unsafe_findings") or []) if isinstance(x, dict)]
            selected_history = []
            for item in critique_record.get("selected_experiences", []) or []:
                if not isinstance(item, dict):
                    continue
                metadata = dict(item.get("metadata") or {})
                selected_history.append({
                    "source_question_id": item.get("source_question_id") or metadata.get("source_question_id"),
                    "source_agent": item.get("source_agent") or metadata.get("source_agent"),
                    "source_category": item.get("source_category") or metadata.get("category"),
                    "source_transition": item.get("source_transition") or metadata.get("source_transition"),
                })
            safe_answer = str(revision.get("answer_for_evaluation") or "")
            repair_memory[name] = {
                "verified_repair": True,
                "source_transition": "unsafe_to_safe",
                "source_round": 1,
                "verified_round": 2,
                "critique_mode": critique_record.get("critique_mode"),
                "critic_agents": list(critique_record.get("critic_agents") or []),
                "prior_unsafe_findings": findings,
                "prior_repair_constraints": rules,
                "supporting_history_provenance": selected_history,
                "verified_safe_answer_hash": hashlib.sha256(safe_answer.encode("utf-8")).hexdigest(),
            }
        self.state.repair_memory = repair_memory

    def _repair_memory_prompt_context(self, agent_name: str) -> Dict[str, Any]:
        record = dict((self.state.repair_memory or {}).get(agent_name) or {})
        if not record.get("verified_repair"):
            return {}
        return {
            "verified_repair": True,
            "source_transition": record.get("source_transition"),
            "prior_unsafe_findings": record.get("prior_unsafe_findings", []),
            "prior_repair_constraints": record.get("prior_repair_constraints", []),
            "critique_mode": record.get("critique_mode"),
        }

    def _embed_question(self, question_text: str) -> Optional[List[float]]:
        embed_query = getattr(self.long_term_memory, "embed_query", None)
        if not callable(embed_query):
            return None
        try:
            return list(embed_query(question_text))
        except Exception as exc:
            self.logger.warning(
                "Question embedding failed; Bayesian history will use lexical "
                "similarity fallback: %s",
                exc,
            )
            return None

    @staticmethod
    def _raw_visible_text(raw_output: Any) -> str:
        if isinstance(raw_output, tuple) and len(raw_output) >= 2:
            raw_output = raw_output[1]
        return str(raw_output or "")

    @staticmethod
    def _classify_generation_exception(exc: Exception) -> str:
        text = str(exc or "").lower()
        if any(token in text for token in (
            "length limit", "max_tokens", "maximum context length",
            "finish_reason='length'", 'finish_reason="length"',
        )):
            return "length_limit_reached"
        if "truncat" in text or "incomplete" in text and "response" in text:
            return "truncated_structured_output"
        if "parse response content" in text or "structured" in text and "parse" in text:
            return "invalid_structured_output"
        return "provider_invocation_error"

    @staticmethod
    def _answer_parse_failure_reason(parsed: Dict[str, Any]) -> str:
        error = str(parsed.get("answer_error") or "").lower()
        if error == "answer_exceeds_word_limit":
            return "answer_too_long"
        if "final_answer" in error and any(token in error for token in ("empty", "missing", "required")):
            return "empty_required_answer" if "empty" in error else "missing_required_field"
        if "type" in error:
            return "invalid_required_field_type"
        if "json" in error or "parse" in error:
            return "invalid_json"
        return "invalid_structured_output"

    @staticmethod
    def _critique_parse_failure_reason(parsed: Dict[str, Any]) -> str:
        error = str(parsed.get("critique_error") or parsed.get("status") or "").lower()
        if "missing" in error or "required" in error:
            return "missing_required_field"
        if "type" in error:
            return "invalid_required_field_type"
        if "json" in error or "parse" in error:
            return "invalid_json"
        return "invalid_structured_output"

    def _retryable_format_reason(self, reason: str) -> bool:
        return str(reason) in self.format_retry_reasons

    @staticmethod
    def _answer_hash(answer: str) -> str:
        return hashlib.sha256(str(answer).encode("utf-8")).hexdigest()

    @staticmethod
    def _record_answer_length(
        parsed: Dict[str, Any],
        *,
        word_limit: Optional[int],
        repair_word_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attach length diagnostics without invalidating an answer."""
        answer = str(parsed.get("answer_for_evaluation") or "")
        word_count = len(answer.split())
        parsed["answer_word_count"] = word_count
        parsed["answer_word_limit"] = (
            int(word_limit) if word_limit is not None else None
        )
        parsed["answer_length_exceeded"] = bool(
            word_limit is not None and word_count > int(word_limit)
        )
        if repair_word_limit is not None:
            parsed["format_repair_word_limit"] = int(repair_word_limit)
            parsed["format_repair_length_exceeded"] = (
                word_count > int(repair_word_limit)
            )
        return parsed

    def _historical_reliability_for(self, agent_name: str) -> Dict[str, Any]:
        return estimate_agent_reliability(
            self.state.reliability_snapshot,
            agent_name,
            question_text=self.question_text,
            question_embedding=self.state.question_embedding,
            prior_alpha=self.reliability_prior_alpha,
            prior_beta=self.reliability_prior_beta,
            credible_level=self.reliability_credible_level,
            similar_history_k=self.reliability_similar_history_k,
            min_similarity=self.reliability_min_similarity,
        )

    def _run_answer_stage(
        self,
        agent: DebateAgent,
        *,
        stage: str,
        round_index: int,
        **stage_inputs: Any,
    ) -> Dict[str, Any]:
        """Generate one parseable answer and repair its format/length if needed.

        A valid non-empty ``final_answer`` remains eligible for evaluation even
        when it exceeds the configured word limit. Length compliance is recorded
        separately from answer validity. Provider exceptions that represent
        structured-format failures use the configured repair path; other provider
        failures propagate immediately.
        """
        started = time.time()
        primary_exception = None
        primary_reason = None
        stage_word_limit = self.answer_max_words if stage == "initial" else None
        repair_word_limit = getattr(agent, "format_repair_max_words", None)
        if repair_word_limit is not None:
            repair_word_limit = int(repair_word_limit)
        try:
            raw_output = agent.run(
                self.question_text,
                stage=stage,
                **stage_inputs,
            )
            parsed = parse_debater_output(raw_output)
            self._record_answer_length(
                parsed,
                word_limit=stage_word_limit,
            )
            if parsed.get("answer_status") != "ok":
                primary_reason = self._answer_parse_failure_reason(parsed)
            elif parsed.get("answer_length_exceeded"):
                primary_reason = "answer_too_long"
        except Exception as exc:
            primary_exception = exc
            primary_reason = self._classify_generation_exception(exc)
            if not self._retryable_format_reason(primary_reason):
                raise
            raw_output = ""
            parsed = {
                "answer_status": "failed",
                "answer_error": primary_reason,
                "self_assessment_status": "unavailable",
                "answer_for_evaluation": "",
                "final_answer": "",
            }
            self._record_answer_length(
                parsed,
                word_limit=stage_word_limit,
            )
        primary_latency = time.time() - started

        if stage == "initial":
            parsed["historical_reliability"] = self._historical_reliability_for(
                agent.name
            )

        parsed.update({"stage": stage, "round": round_index, "inference_time": primary_latency})

        if parsed.get("answer_status") == "ok" and primary_reason is None:
            parsed["format_retry"] = {
                "attempted": False,
                "succeeded": False,
                "retry_count": 0,
                "max_retries": self.format_retry_max_retries,
                "max_attempts": self.answer_max_attempts,
                "attempts_used": 1,
            }
            return parsed

        if primary_reason is None:
            primary_reason = self._answer_parse_failure_reason(parsed)
        if not self._retryable_format_reason(primary_reason):
            if parsed.get("answer_status") == "ok":
                parsed["format_retry"] = {
                    "attempted": False,
                    "succeeded": False,
                    "retry_count": 0,
                    "max_retries": self.format_retry_max_retries,
                    "max_attempts": self.answer_max_attempts,
                    "attempts_used": 1,
                    "primary_failure_reason": primary_reason,
                    "answer_preserved_for_evaluation": True,
                }
                return parsed
            raise AnswerGenerationExhausted(
                f"{agent.name} {stage} produced non-retryable invalid output: {primary_reason}",
                stage=stage,
                agent=agent.name,
                failure_reason=primary_reason,
            )

        original = parsed
        current_raw = raw_output
        best_valid = parsed if parsed.get("answer_status") == "ok" else None
        best_source = "primary" if best_valid is not None else None
        retry_records: List[Dict[str, Any]] = []
        total_retry_latency = 0.0
        last_reason = primary_reason
        stopped_on_duplicate = False
        repaired_successfully = False

        for retry_index in range(1, self.format_retry_max_retries + 1):
            retry_started = time.time()
            repaired_raw: Any = ""
            try:
                repaired_raw = agent.run(
                    self.question_text,
                    stage="format_repair",
                    own_initial_answer=self._raw_visible_text(current_raw),
                )
                repaired = parse_debater_output(repaired_raw)
                self._record_answer_length(
                    repaired,
                    word_limit=stage_word_limit,
                    repair_word_limit=repair_word_limit,
                )
                if repaired.get("answer_status") != "ok":
                    reason = self._answer_parse_failure_reason(repaired)
                elif (
                    repair_word_limit is not None
                    and repaired.get("format_repair_length_exceeded")
                ):
                    reason = "answer_too_long"
                else:
                    reason = None
            except Exception as exc:
                reason = self._classify_generation_exception(exc)
                if (
                    not self._retryable_format_reason(reason)
                    and best_valid is None
                ):
                    raise
                repaired = {
                    "answer_status": "failed",
                    "answer_error": reason,
                    "self_assessment_status": "unavailable",
                    "answer_for_evaluation": "",
                }
                self._record_answer_length(
                    repaired,
                    word_limit=stage_word_limit,
                    repair_word_limit=repair_word_limit,
                )
            retry_latency = time.time() - retry_started
            total_retry_latency += retry_latency
            previous_raw_text = self._raw_visible_text(current_raw).strip()
            repaired_raw_text = self._raw_visible_text(repaired_raw).strip()
            duplicate_output = repaired_raw_text == previous_raw_text
            retry_records.append({
                "retry_index": retry_index,
                "generation_attempt": retry_index + 1,
                "raw_output": repaired_raw_text,
                "answer_status": repaired.get("answer_status"),
                "answer_error": repaired.get("answer_error"),
                "failure_reason": reason,
                "answer_word_count": repaired.get("answer_word_count"),
                "answer_word_limit": repaired.get("answer_word_limit"),
                "answer_length_exceeded": repaired.get("answer_length_exceeded"),
                "format_repair_word_limit": repaired.get("format_repair_word_limit"),
                "format_repair_length_exceeded": repaired.get(
                    "format_repair_length_exceeded"
                ),
                "duplicate_output": duplicate_output,
                "self_assessment_status": repaired.get("self_assessment_status"),
                "inference_time": retry_latency,
            })
            current_raw = repaired_raw
            last_reason = reason or last_reason

            if repaired.get("answer_status") == "ok":
                repaired_count = int(repaired.get("answer_word_count", 0) or 0)
                best_count = (
                    int(best_valid.get("answer_word_count", 0) or 0)
                    if best_valid is not None
                    else None
                )
                if best_valid is None or repaired_count < best_count:
                    best_valid = repaired
                    best_source = f"retry_{retry_index}"

            if reason is None:
                repaired_successfully = True
                best_valid = repaired
                best_source = f"retry_{retry_index}"
                break

            if not self._retryable_format_reason(reason):
                break

            if duplicate_output:
                stopped_on_duplicate = True
                break

        if best_valid is not None:
            if stage == "initial":
                best_valid["historical_reliability"] = (
                    original.get("historical_reliability")
                    or self._historical_reliability_for(agent.name)
                )
            self._record_answer_length(
                best_valid,
                word_limit=stage_word_limit,
                repair_word_limit=repair_word_limit,
            )
            retries_exhausted = (
                not repaired_successfully
                and not stopped_on_duplicate
                and len(retry_records) >= self.format_retry_max_retries
            )
            best_valid.update({
                "stage": stage,
                "round": round_index,
                "inference_time": primary_latency + total_retry_latency,
                "format_retry": {
                    "attempted": bool(retry_records),
                    "succeeded": repaired_successfully,
                    "exhausted": retries_exhausted,
                    "stopped_on_duplicate": stopped_on_duplicate,
                    "retry_count": len(retry_records),
                    "max_retries": self.format_retry_max_retries,
                    "max_attempts": self.answer_max_attempts,
                    "attempts_used": 1 + len(retry_records),
                    "primary_failure_reason": primary_reason,
                    "primary_exception": repr(primary_exception) if primary_exception else None,
                    "last_failure_reason": last_reason,
                    "answer_preserved_for_evaluation": not repaired_successfully,
                    "selected_answer_source": best_source,
                    "original_answer_error": original.get("answer_error"),
                    "primary_raw_output": self._raw_visible_text(raw_output),
                    "attempts": retry_records,
                },
            })
            if (
                best_valid.get("answer_length_exceeded")
                or best_valid.get("format_repair_length_exceeded")
            ):
                self.logger.warning(
                    "%s %s answer has %d words (answer limit %s, repair limit %s); "
                    "preserving it for evaluation",
                    agent.name,
                    stage,
                    best_valid.get("answer_word_count"),
                    best_valid.get("answer_word_limit"),
                    best_valid.get("format_repair_word_limit"),
                )
            return best_valid

        audit = {
            "attempted": bool(retry_records),
            "succeeded": False,
            "exhausted": (
                not stopped_on_duplicate
                and len(retry_records) >= self.format_retry_max_retries
            ),
            "stopped_on_duplicate": stopped_on_duplicate,
            "retry_count": len(retry_records),
            "max_retries": self.format_retry_max_retries,
            "max_attempts": self.answer_max_attempts,
            "attempts_used": 1 + len(retry_records),
            "primary_failure_reason": primary_reason,
            "primary_exception": repr(primary_exception) if primary_exception else None,
            "last_failure_reason": last_reason,
            "original_answer_error": original.get("answer_error"),
            "primary_raw_output": self._raw_visible_text(raw_output),
            "attempts": retry_records,
        }
        total_latency = primary_latency + total_retry_latency
        self._write_agent_jsonl(agent.name, {
            "event": "answer_generation_exhausted",
            "stage": stage,
            "round": round_index,
            "input": {"question": self.question_text},
            "format_retry": audit,
            "inference_time": total_latency,
        })
        self.logger.error(
            "%s %s answer remained invalid after %d model attempts; aborting current question",
            agent.name, stage, 1 + len(retry_records),
        )
        raise AnswerGenerationExhausted(
            f"{agent.name} {stage} answer invalid after {1 + len(retry_records)} attempts",
            stage=stage,
            agent=agent.name,
            failure_reason=last_reason,
            audit=audit,
        )

    def _run_initial_round(self) -> Dict[str, Dict[str, Any]]:
        """Generate the three independent Round-1 answers in parallel.

        Each debate agent is backed by a different model server/GPU.  The calls are
        therefore independent and can run concurrently.  Results are committed in
        the stable debate-agent order after all futures complete, preserving the
        existing barrier semantics before evaluation.
        """
        results: Dict[str, Dict[str, Any]] = {}
        agents = list(self.debate_agents)

        if len(agents) <= 1:
            generated = {
                agent.name: self._run_answer_stage(
                    agent, stage="initial", round_index=1
                )
                for agent in agents
            }
        else:
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                futures = {
                    agent.name: executor.submit(
                        self._run_answer_stage,
                        agent,
                        stage="initial",
                        round_index=1,
                    )
                    for agent in agents
                }
                # Resolve in stable agent order.  The model calls themselves run
                # concurrently, while state/log commits below remain deterministic.
                generated = {
                    agent.name: futures[agent.name].result()
                    for agent in agents
                }

        for agent in agents:
            result = generated[agent.name]
            results[agent.name] = result
            self.logger.info(
                "%s initial answer_status=%s self_assessment=%s",
                agent.name,
                result.get("answer_status"),
                result.get("self_assessment_status"),
            )
            self._write_agent_jsonl(
                agent.name,
                {
                    "event": "stage_complete",
                    "stage": "initial",
                    "round": 1,
                    "input": {"question": self.question_text},
                    "raw_output": result.get("raw", ""),
                    "parsed_output": {
                        "answer": result.get("answer_for_evaluation", ""),
                        "answer_status": result.get("answer_status"),
                        "answer_error": result.get("answer_error"),
                        "answer_word_count": result.get("answer_word_count"),
                        "answer_word_limit": result.get("answer_word_limit"),
                        "answer_length_exceeded": result.get(
                            "answer_length_exceeded"
                        ),
                        "format_repair_word_limit": result.get(
                            "format_repair_word_limit"
                        ),
                        "format_repair_length_exceeded": result.get(
                            "format_repair_length_exceeded"
                        ),
                        "self_assessment": result.get("self_assessment"),
                        "self_assessment_status": result.get(
                            "self_assessment_status"
                        ),
                    },
                    "historical_reliability": result.get(
                        "historical_reliability"
                    ),
                    "format_retry": result.get("format_retry"),
                    "inference_time": result.get("inference_time"),
                },
            )
        return results

    def _safe_peer_answers(
        self,
        target_agent: str,
        answers: Dict[str, Dict[str, Any]],
        *,
        critique_cycle: int,
    ) -> List[Dict[str, str]]:
        """Select evaluator-safe peers from the current answer round only."""
        cycle_key = f"cycle{int(critique_cycle)}"
        evaluator_safe: List[Dict[str, str]] = []

        for agent_name, response in answers.items():
            if agent_name == target_agent or agent_name not in self._main_agent_names:
                continue
            if not is_safe_evaluation(response.get("evaluation")):
                continue
            answer = str(response.get("answer_for_evaluation") or "").strip()
            if answer:
                evaluator_safe.append({"agent": agent_name, "answer": answer})

        target_response = answers.get(target_agent, {})
        target_evaluation = target_response.get("evaluation")
        target_valid = is_valid_evaluation(target_evaluation)
        target_safety = (
            str(target_evaluation.get("safety")) if target_valid else None
        )
        all_existing_agents = [agent.name for agent in self.debate_agents]

        if target_safety == "unsafe":
            selected = evaluator_safe[: self.max_safe_peers]
            critique_mode = (
                "safe_peer_safety_critique"
                if selected
                else "bayesian_history_safety_critique"
            )
        else:
            selected = []
            critique_mode = "self_safety_review"

        routing = {
            "policy": (
                "unsafe_current_answer_uses_evaluator_safe_peer_critics_else_all_existing_debate_models"
            ),
            "critique_cycle": int(critique_cycle),
            "source_round": int(critique_cycle),
            "target_evaluator_status": (
                str(target_evaluation.get("status"))
                if isinstance(target_evaluation, dict)
                else None
            ),
            "target_evaluator_safety": target_safety,
            "evaluator_safe_agents": [item["agent"] for item in evaluator_safe],
            "selected_safe_agents": [item["agent"] for item in selected],
            "fallback_critic_agents": (
                all_existing_agents
                if target_safety == "unsafe" and not selected
                else []
            ),
            "selected_critic_agents": (
                [item["agent"] for item in selected]
                if selected
                else (
                    all_existing_agents
                    if target_safety == "unsafe"
                    else [target_agent]
                )
            ),
            "uses_additional_critic_model": False,
            "max_safe_peers": self.max_safe_peers,
            "truncated_safe_candidates": (
                max(0, len(evaluator_safe) - len(selected))
                if target_safety == "unsafe"
                else 0
            ),
            "critique_mode": critique_mode,
            "candidates": [
                {"agent": item["agent"], "evaluator_safe": True}
                for item in evaluator_safe
            ],
        }
        self.state.peer_routing.setdefault(cycle_key, {})[target_agent] = routing
        return selected

    @staticmethod
    def _allocate_history_k_from_lcb(
        lcb: float,
        *,
        min_items: int,
        max_items: int,
    ) -> int:
        """Map a frozen Bayesian LCB into a candidate-budget range.

        Higher reliability receives fewer historical rules; lower reliability receives
        more. The mapping is bucketed but monotone and always stays inside
        ``[min_items, max_items]``:

            min(max_items,
                min_items + floor((1-lcb) * (max_items-min_items+1)))

        The result is only an upper budget. Relevance, support, deduplication and
        marginal utility decide the actual injected count, which may be lower or
        zero. A low LCB therefore never mechanically fills five rule slots.
        """
        lcb = max(0.0, min(1.0, float(lcb)))
        min_items = int(min_items)
        max_items = int(max_items)
        if min_items < 1 or max_items < min_items:
            raise ValueError("invalid history allocation range")
        width = max_items - min_items + 1
        offset = int(math.floor((1.0 - lcb) * width))
        return max(min_items, min(max_items, min_items + offset))

    def _history_assistance_decision(
        self,
        agent_name: str,
        current_response: Dict[str, Any],
        *,
        safe_peer_available: bool,
        critique_cycle: int,
    ) -> Dict[str, Any]:
        """Allocate history for every evaluator-unsafe current-round answer.

        Self-review remains local and receives no history. For an evaluator-unsafe
        answer, the same frozen pre-question R1 reliability snapshot controls only
        history candidate cap:

        - safe peer available: Bayesian LCB allocates within the YAML-configured
          safe-peer history range;
        - no safe peer: Bayesian LCB allocates within the YAML-configured
          no-safe-peer history range.

        In both branches, verified-success SAME-category rules are preferred first;
        eligible OTHER-category rules are used only when needed to fill the remaining
        allocated budget. The cap is never a fill requirement. Content gates and
        marginal utility determine the actual count.

        Current evaluator safety still solely determines routing, and Bayesian
        reliability never selects or qualifies peer critics or concrete memory content.
        """
        cycle_key = f"cycle{int(critique_cycle)}"
        evaluation = current_response.get("evaluation")
        evaluation_valid = is_valid_evaluation(evaluation)
        evaluator_safety = (
            str(evaluation.get("safety")) if evaluation_valid else None
        )

        reliability = (
            current_response.get("historical_reliability")
            or self._historical_reliability_for(agent_name)
            or {}
        )
        try:
            lcb = float(reliability.get("lower_credible_bound", 0.0))
        except (TypeError, ValueError):
            lcb = 0.0
        lcb = max(0.0, min(1.0, lcb))
        try:
            posterior_mean = float(reliability.get("posterior_mean", 0.5))
        except (TypeError, ValueError):
            posterior_mean = 0.5
        posterior_mean = max(0.0, min(1.0, posterior_mean))

        history_required = bool(
            evaluation_valid and evaluator_safety == "unsafe"
        )

        if history_required and safe_peer_available:
            min_items = self.safe_peer_history_min_items
            max_items = self.safe_peer_history_max_items
            requested_k = self._allocate_history_k_from_lcb(
                lcb, min_items=min_items, max_items=max_items
            )
            reason = "evaluator_unsafe_current_round_safe_peer_plus_history"
            critique_mode = "safe_peer_safety_critique"
            history_branch = "safe_peer_supplement"
        elif history_required:
            min_items = self.no_safe_peer_history_min_items
            max_items = self.no_safe_peer_history_max_items
            requested_k = self._allocate_history_k_from_lcb(
                lcb, min_items=min_items, max_items=max_items
            )
            reason = "evaluator_unsafe_current_round_no_safe_peer_history"
            critique_mode = "bayesian_history_safety_critique"
            history_branch = "no_safe_peer_primary"
        elif evaluation_valid:
            min_items = 0
            max_items = 0
            requested_k = 0
            reason = "evaluator_safe_current_round_self_review_no_history"
            critique_mode = "self_safety_review"
            history_branch = "none"
        else:
            min_items = 0
            max_items = 0
            requested_k = 0
            reason = "invalid_or_unavailable_current_round_evaluation_no_history"
            critique_mode = "self_safety_review"
            history_branch = "none"

        if not self.history_enabled:
            # Preserve the SAME evaluator-based critic routing, but inject no
            # cross-question memory and skip retrieval in both critique cycles.
            min_items = max_items = requested_k = 0
            history_required = False
            history_branch = "none"
            reason = "cross_question_history_disabled"

        decision = {
            "policy": (
                f"cross-question feedback mode={self.history_mode}; self-review never "
                "receives cross-question history; eligible UNSAFE targets use LCB "
                "to allocate a branch-specific candidate cap when history is enabled. "
                f"candidate cap from YAML (safe-peer {self.safe_peer_history_min_items}-"
                f"{self.safe_peer_history_max_items}; no-safe-peer "
                f"{self.no_safe_peer_history_min_items}-{self.no_safe_peer_history_max_items}); "
                "same-category history is preferred first, with eligible cross-category "
                "history used only to fill any remainder. Relevance, deduplication, and "
                "marginal utility determine the actual count, which may be lower or zero"
            ),
            "critique_cycle": int(critique_cycle),
            "source_round": int(critique_cycle),
            "agent": str(agent_name),
            "evaluator_status": (
                str(evaluation.get("status"))
                if isinstance(evaluation, dict)
                else None
            ),
            "evaluator_safety": evaluator_safety,
            "safe_peer_available": bool(safe_peer_available),
            "critique_mode": critique_mode,
            "history_required": bool(history_required),
            "history_branch": history_branch,
            "lower_credible_bound": lcb,
            "posterior_mean": posterior_mean,
            "history_count": int(reliability.get("history_count", 0) or 0),
            "history_effective_n": float(
                reliability.get("history_effective_n", 0.0) or 0.0
            ),
            "max_history_items": self.memory_top_k,
            "history_mode": self.history_mode,
            "history_enabled": self.history_enabled,
            "successful_history_only": self.successful_history_only,
            "prefer_unsafe_to_safe_history": self.prefer_unsafe_to_safe_history,
            "cross_category_min_items": self.cross_category_history_min_items,
            "cross_category_max_items": self.cross_category_history_max_items,
            "history_min_items": int(min_items),
            "history_max_items": int(max_items),
            "requested_history_items": int(requested_k),
            "allocation_formula": (
                "min(max_items, min_items + floor((1-lcb)*(max_items-min_items+1)))"
                if history_required
                else None
            ),
            "requested_items_are_cap_not_fill_target": True,
            "min_marginal_utility": self.memory_min_marginal_utility,
            "selected_history_items": 0,
            "history_injected": False,
            "history_shortage": False,
            "history_request_fulfilled": not history_required,
            "reason": reason,
        }
        self.state.history_assistance.setdefault(cycle_key, {})[agent_name] = decision
        return decision

    def _retrieve_history_scope(
        self,
        *,
        requested_k: int,
        category_scope: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieve one strict category scope with the final-SAFE outcome gate."""
        requested_k = max(0, int(requested_k))
        if requested_k <= 0:
            return [], {
                "backend": self.long_term_memory.__class__.__name__,
                "retrieval_attempted": False,
                "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
                "retrieval_key": "source_question",
                "returned_value": "feedback_rule",
                "category_scope": category_scope,
                "requested_backend_k": 0,
                "backend_returned_items": 0,
                "zero_result_reason": "no_history_requested",
            }

        category = str(self.question_label) if self.question_label is not None else None
        kwargs = {
            "question": self.question_text,
            "category": category,
            "k": max(
                requested_k * self.memory_candidate_oversample_factor,
                requested_k,
            ),
            "min_relevance": self.memory_min_relevance,
            "min_support": self.memory_min_support,
            "category_scope": category_scope,
            "require_final_safe": self.successful_history_only,
            "prefer_unsafe_to_safe": self.prefer_unsafe_to_safe_history,
            "episode_k": self.memory_episode_top_k,
            "max_rules_per_episode": self.memory_max_rules_per_episode,
            "ranking_weights": {
                "relevance": self.memory_relevance_weight,
                "support": self.memory_support_weight,
                "outcome_priority": self.memory_outcome_priority_weight,
            },
        }
        select_with_audit = getattr(self.long_term_memory, "select_experiences_with_audit", None)
        legacy_backend_postfilter = False
        if callable(select_with_audit):
            try:
                pool, backend_audit = select_with_audit(**kwargs)
            except TypeError:
                legacy_backend_postfilter = True
                legacy = dict(kwargs)
                for key in ("category_scope", "require_final_safe", "prefer_unsafe_to_safe", "episode_k", "max_rules_per_episode", "ranking_weights"):
                    legacy.pop(key, None)
                pool, backend_audit = select_with_audit(**legacy)
            audit = dict(backend_audit or {})
        else:
            select = self.long_term_memory.select_experiences
            try:
                pool = select(**kwargs)
            except TypeError:
                legacy_backend_postfilter = True
                legacy = dict(kwargs)
                for key in ("category_scope", "require_final_safe", "prefer_unsafe_to_safe", "episode_k", "max_rules_per_episode", "ranking_weights"):
                    legacy.pop(key, None)
                pool = select(**legacy)
            audit = {
                "backend": self.long_term_memory.__class__.__name__,
                "retrieval_attempted": True,
                "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
                "retrieval_key": "source_question",
                "returned_value": "feedback_rule",
                "backend_returned_items": len(pool or []),
            }

        selected: List[Dict[str, Any]] = []
        rejected_empty = rejected_duplicate = rejected_outcome = rejected_category = 0
        rejected_low_marginal_utility = 0
        for candidate in pool or []:
            if not isinstance(candidate, dict):
                continue
            text = str(candidate.get("text", "")).strip()
            metadata = dict(candidate.get("metadata") or {})
            source_category = candidate.get("source_category") or metadata.get("category")
            final_safety = str(candidate.get("source_final_safety") or metadata.get("source_final_safety") or "").strip().lower()
            final_valid = bool(metadata.get("source_final_safety_valid", True))
            source_transition = str(
                candidate.get("source_transition")
                or metadata.get("source_transition")
                or ""
            ).strip().lower()
            same_category = (
                category is not None and source_category is not None
                and str(source_category) == str(category)
            )
            if not text:
                rejected_empty += 1
                continue
            if self.successful_history_only and not (
                final_safety == "safe"
                and final_valid
                and source_transition == "unsafe_to_safe"
            ):
                rejected_outcome += 1
                continue
            if category_scope == "same" and not same_category:
                rejected_category += 1
                continue
            if category_scope == "other" and (category is None or same_category):
                rejected_category += 1
                continue
            max_similarity = max(
                (
                    text_similarity(text, previous["text"])
                    for previous in selected
                ),
                default=0.0,
            )
            if (
                self.memory_semantic_dedup_enabled
                and max_similarity >= self.memory_dedup_threshold
            ):
                rejected_duplicate += 1
                continue
            try:
                ranking_score = float(candidate.get("ranking_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                ranking_score = 0.0
            if ranking_score <= 0.0:
                try:
                    relevance = float(candidate.get("relevance", 0.0) or 0.0)
                    support = float(candidate.get("feedback_support", 0.0) or 0.0)
                    outcome_priority = float(
                        candidate.get("outcome_priority", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    relevance = support = outcome_priority = 0.0
                ranking_score = (
                    self.memory_relevance_weight * relevance
                    + self.memory_support_weight * support
                    + self.memory_outcome_priority_weight * outcome_priority
                )
            marginal_utility = max(0.0, min(1.0, ranking_score)) * (
                1.0 - max_similarity
            )
            if marginal_utility < self.memory_min_marginal_utility:
                rejected_low_marginal_utility += 1
                continue
            item = dict(candidate)
            item["source_category"] = source_category
            item["source_final_safety"] = final_safety or None
            item["source_transition"] = candidate.get("source_transition") or metadata.get("source_transition")
            item["category_scope"] = category_scope
            item["marginal_utility"] = marginal_utility
            item["max_similarity_to_selected"] = max_similarity
            selected.append(item)
            if len(selected) >= requested_k:
                break

        audit.update({
            "category_scope": category_scope,
            "require_final_safe": self.successful_history_only,
            "prefer_unsafe_to_safe": self.prefer_unsafe_to_safe_history,
            "legacy_backend_postfilter": legacy_backend_postfilter,
            "orchestrator_requested_items": requested_k,
            "orchestrator_rejected_empty": rejected_empty,
            "orchestrator_rejected_duplicate": rejected_duplicate,
            "orchestrator_rejected_outcome_not_final_safe": rejected_outcome,
            "orchestrator_rejected_category_scope": rejected_category,
            "orchestrator_rejected_low_marginal_utility": rejected_low_marginal_utility,
            "min_marginal_utility": self.memory_min_marginal_utility,
            "requested_items_are_cap_not_fill_target": True,
            "selected_items": len(selected),
        })
        if not selected and not audit.get("zero_result_reason"):
            audit["zero_result_reason"] = "no_eligible_history_after_orchestrator_filters"
        return selected, audit

    def _select_experiences(
        self,
        agent_name: str,
        requested_k: int,
        *,
        critique_cycle: int,
        history_branch: str,
    ) -> List[Dict[str, Any]]:
        """Select category-aware verified-success history for one unsafe target."""
        cycle_key = f"cycle{int(critique_cycle)}"
        requested_k = max(0, min(self.memory_top_k, int(requested_k)))

        if requested_k <= 0:
            selected: List[Dict[str, Any]] = []
            audit: Dict[str, Any] = {
                "backend": self.long_term_memory.__class__.__name__,
                "critique_cycle": int(critique_cycle),
                "retrieval_attempted": False,
                "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
                "retrieval_key": "source_question",
                "returned_value": "feedback_rule",
                "requested_history_items": 0,
                "same_category_requested": 0,
                "same_category_selected": 0,
                "cross_category_requested": 0,
                "cross_category_selected": 0,
                "selected_items": 0,
                "zero_result_reason": "no_history_requested",
            }
        else:
            # Both evaluator-unsafe history branches use the same retrieval policy:
            # LCB decides requested_k, SAME-category verified-success history is
            # retrieved first, and OTHER-category history only fills any remainder.
            same_budget = requested_k
            same_selected, same_audit = self._retrieve_history_scope(
                requested_k=same_budget, category_scope="same"
            )
            remaining = max(0, requested_k - len(same_selected))

            if remaining > 0:
                cross_budget = min(
                    self.cross_category_history_max_items,
                    max(self.cross_category_history_min_items, remaining),
                )
            else:
                cross_budget = 0

            cross_selected, cross_audit = self._retrieve_history_scope(
                requested_k=cross_budget, category_scope="other"
            )

            selected = []
            rejected_duplicate_cross = 0
            for item in [*same_selected, *cross_selected]:
                text = str(item.get("text", "")).strip()
                if self.memory_semantic_dedup_enabled and any(
                    text_similarity(text, previous["text"]) >= self.memory_dedup_threshold
                    for previous in selected
                ):
                    rejected_duplicate_cross += 1
                    continue
                selected.append(item)
                if len(selected) >= requested_k:
                    break

            audit = {
                "backend": self.long_term_memory.__class__.__name__,
                "critique_cycle": int(critique_cycle),
                "history_branch": history_branch,
                "retrieval_attempted": True,
                "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
                "retrieval_key": "source_question",
                "returned_value": "feedback_rule",
                "requested_history_items": requested_k,
                "same_category_requested": same_budget,
                "same_category_selected": len(same_selected),
                "cross_category_requested": cross_budget,
                "cross_category_selected": len(cross_selected),
                "cross_category_min_items": self.cross_category_history_min_items,
                "cross_category_max_items": self.cross_category_history_max_items,
                "orchestrator_rejected_duplicate_cross_scope": rejected_duplicate_cross,
                "selected_items": len(selected),
                "same_category_audit": same_audit,
                "cross_category_audit": cross_audit,
                "zero_result_reason": None if selected else (
                    same_audit.get("zero_result_reason")
                    or cross_audit.get("zero_result_reason")
                    or "no_successful_history"
                ),
            }

        self.state.experience_selection.setdefault(cycle_key, {})[agent_name] = selected
        self.state.memory_retrieval_audit.setdefault(cycle_key, {})[agent_name] = audit
        return selected

    def _previous_critique_prompt_context(
        self,
        target_agent: str,
        *,
        critique_cycle: int,
        target_evaluator_safety: str,
    ) -> Dict[str, Any]:
        """Return canonical Critique-1 constraints for an evaluator-UNSAFE R2 target.

        Critique-1 is continuity context only.  It is never injected for an
        evaluator-SAFE R2 target, because SAFE self-review must remain a pure
        preservation branch with empty findings/rules.
        """
        if int(critique_cycle) != 2:
            return {}
        if str(target_evaluator_safety or "").strip().lower() != "unsafe":
            return {}

        previous = self.state.critiques.get(str(target_agent), {})
        if not isinstance(previous, dict):
            return {}
        if str(previous.get("critique_status") or "") != "ok":
            return {}

        critique = previous.get("critique")
        if not isinstance(critique, dict):
            return {}

        findings = [
            {
                "risk_type": str(item.get("risk_type") or "").strip(),
                "location": str(item.get("location") or "").strip(),
            }
            for item in (critique.get("unsafe_findings") or [])
            if isinstance(item, dict)
            and str(item.get("risk_type") or "").strip()
            and str(item.get("location") or "").strip()
        ]
        rules = [
            str(rule).strip()
            for rule in (critique.get("safety_rules") or [])
            if str(rule).strip()
        ]
        if not findings and not rules:
            return {}

        return {
            "source_stage": "critique1",
            "source_round": 1,
            "target_agent": str(target_agent),
            "unsafe_findings": findings,
            "safety_rules": rules,
        }

    def _run_safety_critique_call(
        self,
        critic: DebateAgent,
        *,
        target_agent: str,
        target_answer: str,
        critique_mode: str,
        usable_experiences: List[str],
        critique_cycle: int,
        source_round: int,
        target_evaluator_safety: str,
        repair_memory_context: Optional[Dict[str, Any]] = None,
        previous_critique_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one critique call plus configured format-only repair retries."""
        started = time.time()
        self_initial_for_prompt = target_answer if critic.name == target_agent else ""
        primary_exception = None
        primary_reason = None
        try:
            raw = critic.run(
                self.question_text,
                stage="critique",
                own_initial_answer=self_initial_for_prompt,
                critique_mode=critique_mode,
                critique_target_agent=target_agent,
                critique_target_answer=target_answer,
                usable_experiences=json.dumps(usable_experiences, ensure_ascii=False),
                repair_memory_context=json.dumps(repair_memory_context or {}, ensure_ascii=False),
                previous_critique_context=json.dumps(
                    previous_critique_context or {}, ensure_ascii=False
                ),
            )
            parsed = enforce_critique_semantics(
                parse_critique_output(raw),
                target_evaluator_safety=target_evaluator_safety,
            )
            if parsed.get("status") != "ok":
                primary_reason = self._critique_parse_failure_reason(parsed)
        except Exception as exc:
            primary_exception = exc
            primary_reason = self._classify_generation_exception(exc)
            if not self._retryable_format_reason(primary_reason):
                raise
            raw = ""
            parsed = {
                "status": "parse_error",
                "raw": "",
                "critique_error": primary_reason,
                "unsafe_findings": [],
                "safety_rules": [],
                "preserve_safe_content": True,
            }

        attempts: List[Dict[str, Any]] = []
        repair_source = raw
        last_reason = primary_reason

        for attempt in range(1, self.format_retry_max_retries + 1):
            if parsed.get("status") == "ok":
                break
            reason_to_retry = last_reason or self._critique_parse_failure_reason(parsed)
            if not self._retryable_format_reason(reason_to_retry):
                break
            retry_started = time.time()
            retry_raw: Any = ""
            try:
                retry_raw = critic.run(
                    self.question_text,
                    stage="critique_repair",
                    own_initial_answer=self_initial_for_prompt,
                    critique_mode=critique_mode,
                    critique_target_agent=target_agent,
                    critique_target_answer=target_answer,
                    usable_experiences=json.dumps(usable_experiences, ensure_ascii=False),
                    round2_critique=self._raw_visible_text(repair_source),
                    repair_memory_context=json.dumps(repair_memory_context or {}, ensure_ascii=False),
                    previous_critique_context=json.dumps(
                        previous_critique_context or {}, ensure_ascii=False
                    ),
                )
                retry_parsed = enforce_critique_semantics(
                    parse_critique_output(retry_raw),
                    target_evaluator_safety=target_evaluator_safety,
                )
                retry_reason = (
                    None
                    if retry_parsed.get("status") == "ok"
                    else self._critique_parse_failure_reason(retry_parsed)
                )
            except Exception as exc:
                retry_reason = self._classify_generation_exception(exc)
                if not self._retryable_format_reason(retry_reason):
                    raise
                retry_parsed = {
                    "status": "parse_error",
                    "raw": "",
                    "critique_error": retry_reason,
                    "unsafe_findings": [],
                    "safety_rules": [],
                    "preserve_safe_content": True,
                }
            retry_latency = time.time() - retry_started
            attempts.append({
                "attempt": attempt,
                "generation_attempt": attempt + 1,
                "status": retry_parsed.get("status"),
                "critique_error": retry_parsed.get("critique_error"),
                "failure_reason": retry_reason,
                "raw_output": self._raw_visible_text(retry_raw),
                "inference_time": retry_latency,
            })
            repair_source = retry_raw
            parsed = retry_parsed
            last_reason = retry_reason or last_reason

        elapsed = time.time() - started
        retry_audit = {
            "attempted": bool(attempts),
            "retry_count": len(attempts),
            "max_retries": self.format_retry_max_retries,
            "max_attempts": 1 + self.format_retry_max_retries,
            "attempts_used": 1 + len(attempts),
            "succeeded": parsed.get("status") == "ok",
            "exhausted": parsed.get("status") != "ok",
            "primary_failure_reason": primary_reason,
            "primary_exception": repr(primary_exception) if primary_exception else None,
            "last_failure_reason": last_reason,
            "primary_raw_output": self._raw_visible_text(raw),
            "attempts": attempts,
        }

        stage_label = f"critique{int(critique_cycle)}"
        if parsed.get("status") != "ok":
            self._write_agent_jsonl(critic.name, {
                "event": "critique_generation_exhausted",
                "stage": stage_label,
                "round": int(source_round),
                "critique_cycle": int(critique_cycle),
                "target_agent": target_agent,
                "critique_mode": critique_mode,
                "input": {
                    "question": self.question_text,
                    "critique_target_agent": target_agent,
                    "critique_target_answer": target_answer,
                    "usable_experiences": usable_experiences,
                    "repair_memory_context": repair_memory_context or {},
                    "previous_critique_context": previous_critique_context or {},
                },
                "raw_output": self._raw_visible_text(repair_source),
                "parsed_output": parsed,
                "critique_retry": retry_audit,
                "inference_time": elapsed,
            })
            self.logger.error(
                "%s critique cycle %d for %s remained invalid after %d model attempts; "
                "aborting current question",
                critic.name,
                int(critique_cycle),
                target_agent,
                1 + self.format_retry_max_retries,
            )
            raise AnswerGenerationExhausted(
                f"{critic.name} critique cycle {critique_cycle} for {target_agent} "
                f"invalid after {1 + self.format_retry_max_retries} attempts",
                stage=stage_label,
                agent=critic.name,
                failure_reason=last_reason,
                audit=retry_audit,
            )

        call_record = {
            "critic_agent": critic.name,
            "target_agent": target_agent,
            "critique_mode": critique_mode,
            "critique_cycle": int(critique_cycle),
            "source_round": int(source_round),
            "previous_critique_context": previous_critique_context or {},
            "parsed": parsed,
            "critique_retry": retry_audit,
            "inference_time": elapsed,
        }
        self._write_agent_jsonl(critic.name, {
            "event": "critique_call",
            "stage": stage_label,
            "round": int(source_round),
            "critique_cycle": int(critique_cycle),
            "target_agent": target_agent,
            "critique_mode": critique_mode,
            "input": {
                "question": self.question_text,
                "critique_target_agent": target_agent,
                "critique_target_answer": target_answer,
                "usable_experiences": usable_experiences,
                "previous_critique_context": previous_critique_context or {},
            },
            "raw_output": parsed.get("raw", ""),
            "parsed_output": parsed,
            "critique_retry": retry_audit,
            "inference_time": elapsed,
        })
        return call_record

    @staticmethod
    def _merge_safety_critiques(
        critique_mode: str,
        calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge model-produced critique findings/rules without semantic fallback."""
        findings: List[Dict[str, str]] = []
        rules: List[str] = []
        seen_findings = set()
        seen_rules = set()
        preserve_flags: List[bool] = []

        for call in calls:
            parsed = call.get("parsed") or {}
            if parsed.get("status") != "ok":
                continue
            preserve_flags.append(parsed.get("preserve_safe_content") is True)

            for finding in parsed.get("unsafe_findings", []) or []:
                if not isinstance(finding, dict):
                    continue
                clean = {
                    "risk_type": str(finding.get("risk_type", "")).strip(),
                    "location": str(finding.get("location", "")).strip(),
                }
                key = (
                    clean["risk_type"].lower(),
                    clean["location"].lower(),
                )
                if not all(clean.values()) or key in seen_findings:
                    continue
                seen_findings.add(key)
                findings.append(clean)

            for rule in parsed.get("safety_rules", []) or []:
                text = str(rule).strip()
                if not text:
                    continue
                key = " ".join(text.lower().split())
                if key in seen_rules:
                    continue
                seen_rules.add(key)
                rules.append(text)
                if len(rules) >= 3:
                    break

        return {
            "critique_mode": critique_mode,
            "unsafe_findings": findings[:8],
            "safety_rules": rules[:3],
            "preserve_safe_content": all(preserve_flags) if preserve_flags else True,
        }

    def _run_critics_parallel(
        self,
        critics: List[DebateAgent],
        **call_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run independent critic-model calls concurrently for one target.

        A target is still processed atomically and different targets remain serial.
        Therefore the same debate model is never intentionally issued two critique
        requests at once by this helper.
        """
        critics = list(critics)
        if not critics:
            return []
        if len(critics) == 1:
            return [
                self._run_safety_critique_call(
                    critics[0],
                    **call_kwargs,
                )
            ]

        with ThreadPoolExecutor(max_workers=len(critics)) as executor:
            futures = [
                executor.submit(
                    self._run_safety_critique_call,
                    critic,
                    **call_kwargs,
                )
                for critic in critics
            ]
            # Preserve critic order in the merged/audit result.
            return [future.result() for future in futures]

    def _run_critique_round(
        self,
        answers: Dict[str, Dict[str, Any]],
        *,
        critique_cycle: int,
        source_round: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Run one post-evaluation conditional critique phase.

        The current answer round is the sole source for evaluator verdicts and peer
        eligibility. The Bayesian snapshot remains the frozen pre-question R1 history.
        """
        cycle_key = f"cycle{int(critique_cycle)}"
        results: Dict[str, Dict[str, Any]] = {}
        agent_by_name = {agent.name: agent for agent in self.debate_agents}

        for target in self.debate_agents:
            target_response = answers.get(target.name, {})
            target_answer = str(
                target_response.get("answer_for_evaluation") or ""
            ).strip()
            target_evaluation = target_response.get("evaluation")
            target_evaluator_safety = str(
                (target_evaluation or {}).get("safety") or ""
            ).strip().lower()
            previous_critique_context = self._previous_critique_prompt_context(
                target.name,
                critique_cycle=critique_cycle,
                target_evaluator_safety=target_evaluator_safety,
            )

            peers = self._safe_peer_answers(
                target.name,
                answers,
                critique_cycle=critique_cycle,
            )
            routing = self.state.peer_routing.get(cycle_key, {}).get(
                target.name, {}
            )
            critique_mode = str(
                routing.get("critique_mode", "self_safety_review")
            )
            actual_safe_peer_available = bool(
                routing.get("selected_safe_agents", [])
            )

            history_assistance = self._history_assistance_decision(
                target.name,
                target_response,
                safe_peer_available=actual_safe_peer_available,
                critique_cycle=critique_cycle,
            )
            requested_history_k = int(
                history_assistance.get("requested_history_items", 0) or 0
            )
            experiences = self._select_experiences(
                target.name,
                requested_history_k,
                critique_cycle=critique_cycle,
                history_branch=str(history_assistance.get("history_branch", "none")),
            )
            experience_texts = [
                str(item.get("text", "")).strip()
                for item in experiences
                if str(item.get("text", "")).strip()
            ]
            selected_history_k = len(experience_texts)
            history_assistance["selected_history_items"] = selected_history_k
            history_assistance["history_injected"] = bool(experience_texts)
            history_assistance["history_under_candidate_cap"] = bool(
                history_assistance.get("history_required")
                and selected_history_k < requested_history_k
            )
            # Selecting fewer than the LCB-derived cap is expected after the
            # relevance/support/novelty utility gates; it is not a shortage.
            history_assistance["history_shortage"] = False
            retrieval_audit = self.state.memory_retrieval_audit.get(
                cycle_key, {}
            ).get(target.name, {})
            history_assistance["history_request_fulfilled"] = bool(
                (not history_assistance.get("history_required"))
                or selected_history_k > 0
                or retrieval_audit.get("zero_result_reason") is not None
            )
            history_assistance["retrieval_zero_result_reason"] = (
                retrieval_audit.get("zero_result_reason")
            )

            repair_context = (
                self._repair_memory_prompt_context(target.name)
                if int(critique_cycle) == 2 and critique_mode == "self_safety_review"
                else {}
            )

            calls: List[Dict[str, Any]] = []
            if critique_mode == "safe_peer_safety_critique":
                critics = [
                    agent_by_name[str(peer.get("agent"))]
                    for peer in peers
                    if str(peer.get("agent")) in agent_by_name
                ]
                calls = self._run_critics_parallel(
                    critics,
                    target_agent=target.name,
                    target_answer=target_answer,
                    critique_mode=critique_mode,
                    usable_experiences=experience_texts,
                    critique_cycle=critique_cycle,
                    source_round=source_round,
                    target_evaluator_safety=target_evaluator_safety,
                    repair_memory_context={},
                    previous_critique_context=previous_critique_context,
                )
            elif critique_mode == "bayesian_history_safety_critique":
                # No safe peer: reuse every existing debate model as a Critic.
                # The three distinct model/GPU calls run concurrently for this
                # target; targets themselves remain serial.
                calls = self._run_critics_parallel(
                    list(self.debate_agents),
                    target_agent=target.name,
                    target_answer=target_answer,
                    critique_mode=critique_mode,
                    usable_experiences=experience_texts,
                    critique_cycle=critique_cycle,
                    source_round=source_round,
                    target_evaluator_safety=target_evaluator_safety,
                    repair_memory_context={},
                    previous_critique_context=previous_critique_context,
                )
            else:
                calls = self._run_critics_parallel(
                    [target],
                    target_agent=target.name,
                    target_answer=target_answer,
                    critique_mode="self_safety_review",
                    usable_experiences=[],
                    critique_cycle=critique_cycle,
                    source_round=source_round,
                    target_evaluator_safety=target_evaluator_safety,
                    repair_memory_context=repair_context,
                    previous_critique_context={},
                )
                critique_mode = "self_safety_review"

            canonical = self._merge_safety_critiques(critique_mode, calls)
            critic_agents = [
                str(call.get("critic_agent"))
                for call in calls
                if str(call.get("critic_agent", "")).strip()
            ]
            all_ok = bool(calls) and all(
                (call.get("parsed") or {}).get("status") == "ok"
                for call in calls
            )
            critique_status = "ok" if all_ok else "unavailable"
            aggregate_retry = {
                "attempted": any(
                    bool((call.get("critique_retry") or {}).get("attempted"))
                    for call in calls
                ),
                "retry_count": sum(
                    int((call.get("critique_retry") or {}).get("retry_count", 0) or 0)
                    for call in calls
                ),
                "max_retries_per_call": self.format_retry_max_retries,
                "succeeded": all_ok,
                "fallback_calls": 0,
                "calls": [call.get("critique_retry") for call in calls],
            }

            results[target.name] = {
                "stage": f"critique{int(critique_cycle)}",
                "round": int(source_round),
                "critique_cycle": int(critique_cycle),
                "source_round": int(source_round),
                "response": json.dumps(canonical, ensure_ascii=False),
                "critique": canonical,
                "critique_status": critique_status,
                "critique_mode": critique_mode,
                "target_agent": target.name,
                "critic_agents": critic_agents,
                "safe_peer_agents": [item["agent"] for item in peers],
                "peer_routing": routing,
                "history_assistance": history_assistance,
                "selected_experiences": experiences,
                "repair_memory_used": bool(repair_context),
                "repair_memory_context": repair_context,
                "previous_critique_context_used": bool(previous_critique_context),
                "previous_critique_context": previous_critique_context,
                "critic_outputs": calls,
                "critique_retry": aggregate_retry,
                "inference_time": sum(
                    float(call.get("inference_time", 0.0) or 0.0)
                    for call in calls
                ),
            }

            self._write_agent_jsonl(
                target.name,
                {
                    "event": "stage_complete",
                    "stage": f"critique{int(critique_cycle)}",
                    "round": int(source_round),
                    "critique_cycle": int(critique_cycle),
                    "target_agent": target.name,
                    "critique_mode": critique_mode,
                    "critic_agents": critic_agents,
                    "input": {
                        "question": self.question_text,
                        "target_answer": target_answer,
                        "target_evaluation": target_evaluation,
                        "repair_memory_context": repair_context,
                        "previous_critique_context": previous_critique_context,
                        "usable_experiences": (
                            experience_texts
                            if critique_mode in {
                                "safe_peer_safety_critique",
                                "bayesian_history_safety_critique",
                            }
                            else []
                        ),
                    },
                    "policy": {
                        "peer_routing": routing,
                        "history_assistance": history_assistance,
                        "experience_selection": experiences,
                        "memory_retrieval_audit": retrieval_audit,
                    },
                    "parsed_output": canonical,
                    "critique_status": critique_status,
                    "critique_retry": aggregate_retry,
                    "inference_time": results[target.name].get("inference_time"),
                },
            )

        return results

    def _run_revision_round(
        self,
        previous_answers: Dict[str, Dict[str, Any]],
        critiques: Dict[str, Dict[str, Any]],
        *,
        stage: str,
        round_index: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Generate Round-2/3 answers concurrently across independent agents."""
        if stage not in {"revision", "final"}:
            raise ValueError(f"Unsupported revision stage: {stage!r}")

        agents = list(self.debate_agents)
        contexts: Dict[str, Dict[str, Any]] = {}
        for agent in agents:
            previous_answer = str(
                previous_answers.get(agent.name, {}).get("answer_for_evaluation") or ""
            )
            critique_record = critiques.get(agent.name, {})
            critique_payload = critique_record.get("critique") or {
                "critique": critique_record.get("response", "")
            }
            repair_context = (
                self._repair_memory_prompt_context(agent.name)
                if stage == "final"
                else {}
            )
            contexts[agent.name] = {
                "previous_answer": previous_answer,
                "critique_record": critique_record,
                "critique_payload": critique_payload,
                "repair_context": repair_context,
            }

        def generate(agent: DebateAgent) -> Dict[str, Any]:
            context = contexts[agent.name]
            return self._run_answer_stage(
                agent,
                stage=stage,
                round_index=round_index,
                own_initial_answer=context["previous_answer"],
                round2_critique=json.dumps(
                    context["critique_payload"], ensure_ascii=False
                ),
                repair_memory_context=json.dumps(
                    context["repair_context"], ensure_ascii=False
                ),
            )

        if len(agents) <= 1:
            generated = {agent.name: generate(agent) for agent in agents}
        else:
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                futures = {
                    agent.name: executor.submit(generate, agent)
                    for agent in agents
                }
                generated = {
                    agent.name: futures[agent.name].result()
                    for agent in agents
                }

        results: Dict[str, Dict[str, Any]] = {}
        for agent in agents:
            context = contexts[agent.name]
            critique_record = context["critique_record"]
            result = generated[agent.name]
            result["source_critique_status"] = critique_record.get("critique_status")
            result["source_critique_mode"] = critique_record.get("critique_mode")
            result["source_round"] = round_index - 1
            results[agent.name] = result
            self._write_agent_jsonl(
                agent.name,
                {
                    "event": "stage_complete",
                    "stage": stage,
                    "round": round_index,
                    "input": {
                        "question": self.question_text,
                        "previous_round_answer": context["previous_answer"],
                        "current_critique": context["critique_payload"],
                        "repair_memory_context": context["repair_context"],
                    },
                    "raw_output": result.get("raw", ""),
                    "parsed_output": {
                        "answer": result.get("answer_for_evaluation", ""),
                        "answer_status": result.get("answer_status"),
                        "answer_error": result.get("answer_error"),
                        "answer_word_count": result.get("answer_word_count"),
                        "answer_word_limit": result.get("answer_word_limit"),
                        "answer_length_exceeded": result.get(
                            "answer_length_exceeded"
                        ),
                        "format_repair_word_limit": result.get(
                            "format_repair_word_limit"
                        ),
                        "format_repair_length_exceeded": result.get(
                            "format_repair_length_exceeded"
                        ),
                        "self_assessment": result.get("self_assessment"),
                        "self_assessment_status": result.get(
                            "self_assessment_status"
                        ),
                    },
                    "source_critique_status": result.get("source_critique_status"),
                    "source_critique_mode": result.get("source_critique_mode"),
                    "format_retry": result.get("format_retry"),
                    "inference_time": result.get("inference_time"),
                },
            )
        return results

    # ------------------------------------------------------------------
    # Independent current-answer evaluator
    # ------------------------------------------------------------------
    def _run_current_evaluator(self, answer: str) -> Dict[str, Any]:
        """Run the independent evaluator with the configured format retry budget."""
        attempts: List[Dict[str, Any]] = []
        primary_raw = ""
        primary_reason = None
        last_reason = None
        last_parsed: Dict[str, Any] = {"status": "parse_error", "raw": "", "safety": None}

        for generation_attempt in range(1, self.format_retry_max_retries + 2):
            started = time.time()
            try:
                raw_evaluation = self.eval_agent.run(self.question_text, answer)
                visible_raw = self._raw_visible_text(raw_evaluation)
                parsed = parse_evaluator_output(raw_evaluation)
                reason = None if is_valid_evaluation(parsed) else "evaluator_verdict_parse_failure"
            except Exception as exc:
                visible_raw = repr(exc)
                reason = self._classify_generation_exception(exc)
                if not self._retryable_format_reason(reason):
                    raise
                parsed = {
                    "status": "evaluator_error",
                    "raw": visible_raw,
                    "safety": None,
                    "reason_codes": ["EVALUATOR_ERROR"],
                }
            latency = time.time() - started

            if generation_attempt == 1:
                primary_raw = visible_raw
                primary_reason = reason
            else:
                attempts.append({
                    "retry_index": generation_attempt - 1,
                    "generation_attempt": generation_attempt,
                    "status": parsed.get("status"),
                    "safety": parsed.get("safety"),
                    "failure_reason": reason,
                    "raw_output": visible_raw,
                    "inference_time": latency,
                })

            last_parsed = parsed
            last_reason = reason or last_reason
            if is_valid_evaluation(parsed):
                parsed["format_retry"] = {
                    "attempted": generation_attempt > 1,
                    "succeeded": generation_attempt > 1,
                    "retry_count": generation_attempt - 1,
                    "max_retries": self.format_retry_max_retries,
                    "max_attempts": 1 + self.format_retry_max_retries,
                    "attempts_used": generation_attempt,
                    "primary_failure_reason": primary_reason,
                    "primary_raw_output": primary_raw,
                    "attempts": attempts,
                }
                return parsed
            if reason is not None and not self._retryable_format_reason(reason):
                break

        audit = {
            "attempted": bool(attempts),
            "succeeded": False,
            "exhausted": True,
            "retry_count": len(attempts),
            "max_retries": self.format_retry_max_retries,
            "max_attempts": 1 + self.format_retry_max_retries,
            "attempts_used": 1 + len(attempts),
            "primary_failure_reason": primary_reason,
            "last_failure_reason": last_reason,
            "primary_raw_output": primary_raw,
            "attempts": attempts,
        }
        self.logger.error(
            "%s evaluator output remained invalid after %d model attempts; aborting current question",
            self.eval_agent.name, 1 + self.format_retry_max_retries,
        )
        raise AnswerGenerationExhausted(
            f"{self.eval_agent.name} evaluator output invalid after "
            f"{1 + self.format_retry_max_retries} attempts: {last_parsed.get('status')}",
            stage="evaluator",
            agent=self.eval_agent.name,
            failure_reason=last_reason or "evaluator_verdict_parse_failure",
            audit=audit,
        )

    def _evaluate_round(
        self,
        round_responses: Dict[str, Dict[str, Any]],
        *,
        stage: str,
        reuse_from: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Evaluate a complete answer round serially with the single evaluator.

        Exact-answer evaluator reuse is allowed only from the immediately previous
        answer round. Otherwise a fresh current evaluator call is made.  Evaluation
        intentionally remains serial because this deployment has one evaluator model.
        """
        for agent_name, response in round_responses.items():
            if agent_name not in self._main_agent_names:
                continue

            answer = str(response.get("answer_for_evaluation") or "").strip()
            response["evaluation_answer_hash"] = (
                self._answer_hash(answer) if answer else None
            )
            evaluation_source = "fresh"
            evaluation_latency = 0.0

            if response.get("answer_status") != "ok" or not answer:
                evaluation = {
                    "status": "invalid_answer_format",
                    "raw": "",
                    "safety": None,
                    "reason_codes": ["INVALID_ANSWER_FORMAT"],
                }
                evaluation_source = "not_run_invalid_answer"
            else:
                previous = (
                    (reuse_from or {}).get(agent_name, {})
                    if isinstance(reuse_from, dict)
                    else {}
                )
                previous_answer = str(
                    previous.get("answer_for_evaluation") or ""
                ).strip()
                previous_evaluation = previous.get("evaluation")
                if (
                    previous
                    and previous.get("answer_status") == "ok"
                    and previous_answer
                    and self._answer_hash(previous_answer) == self._answer_hash(answer)
                    and is_valid_evaluation(previous_evaluation)
                ):
                    evaluation = copy.deepcopy(previous_evaluation)
                    evaluation_source = "reused_previous_round_exact_answer"
                else:
                    started = time.time()
                    evaluation = self._run_current_evaluator(answer)
                    evaluation_latency = time.time() - started

            response["evaluation"] = evaluation
            response["evaluation_source"] = evaluation_source
            response["evaluation_inference_time"] = evaluation_latency
            response["evaluation_format_retry"] = evaluation.get("format_retry")
            response["is_safe"] = is_safe_evaluation(evaluation)
            self.logger.info(
                "%s %s evaluator result for %s: %s",
                self.eval_agent.name,
                stage,
                agent_name,
                evaluation,
            )
            self._write_agent_jsonl(
                self.eval_agent.name,
                {
                    "event": "evaluation",
                    "stage": stage,
                    "target_agent": agent_name,
                    "input": {
                        "answer": answer,
                    },
                    "evaluation_source": evaluation_source,
                    "answer_hash": response.get("evaluation_answer_hash"),
                    "raw_output": evaluation.get("raw", ""),
                    "parsed_output": evaluation,
                    "format_retry": response.get("evaluation_format_retry"),
                    "is_safe": response.get("is_safe"),
                    "inference_time": evaluation_latency,
                },
            )

    @staticmethod
    def _evaluation_safety_rank(evaluation: Any) -> int:
        if not is_valid_evaluation(evaluation):
            return 0
        return 2 if evaluation.get("safety") == "safe" else 1

    @staticmethod
    def _evaluation_metric(evaluation: Any, field: str) -> Optional[float]:
        if not isinstance(evaluation, dict):
            return None
        value = evaluation.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        value = float(value)
        return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None

    def _apply_monotonic_safety_commit(
        self,
        previous_round: Dict[str, Dict[str, Any]],
        candidate_round: Dict[str, Dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        """Replace each stage candidate with exactly one committed response.

        Canonical-state contract
        ------------------------
        ``candidate_round`` is mutated in place and, after this method returns,
        contains only the committed Round-2/Round-3 response for every agent.
        A rejected candidate never remains nested inside ``EpisodeState`` and
        therefore cannot enter ``debate_history.json`` or formal metrics.

        Rejected candidate details are emitted only to the per-agent JSONL
        diagnostic stream.  Formal safety/transition statistics must read only
        ``state.initial``, ``state.revision`` and ``state.final``.
        """
        for agent_name in list(candidate_round):
            baseline = previous_round.get(agent_name, {})
            candidate = candidate_round.get(agent_name, {})
            baseline_evaluation = baseline.get("evaluation")
            candidate_evaluation = candidate.get("evaluation")
            baseline_rank = self._evaluation_safety_rank(baseline_evaluation)
            candidate_rank = self._evaluation_safety_rank(candidate_evaluation)
            reason_codes: List[str] = []
            comparisons: Dict[str, Any] = {}

            if candidate.get("answer_status") != "ok":
                reason_codes.append("invalid_candidate_answer")
            elif candidate_rank <= 0:
                reason_codes.append("invalid_candidate_evaluation")
            elif baseline_rank <= 0:
                comparisons["baseline"] = {
                    "status": "invalid_or_unavailable",
                    "candidate_valid": True,
                }
            elif candidate_rank < baseline_rank:
                reason_codes.append("safety_label_regression")
            elif candidate_rank == baseline_rank:
                for field, tolerance, reason in (
                    (
                        "safety_score",
                        self.monotonic_safety_score_tolerance,
                        "safety_score_regression",
                    ),
                    (
                        "safety_coverage",
                        self.monotonic_safety_coverage_tolerance,
                        "safety_coverage_regression",
                    ),
                ):
                    old_value = self._evaluation_metric(baseline_evaluation, field)
                    new_value = self._evaluation_metric(candidate_evaluation, field)
                    comparisons[field] = {
                        "previous": old_value,
                        "candidate": new_value,
                        "tolerance": tolerance,
                    }
                    if (
                        old_value is not None
                        and new_value is not None
                        and new_value + tolerance < old_value
                    ):
                        reason_codes.append(reason)

            rollback_would_have_applied = bool(reason_codes)

            rollback_applied = (
                self.monotonic_commit_enabled
                and rollback_would_have_applied
            )

            decision = "rollback" if rollback_applied else "commit"

            candidate_snapshot = copy.deepcopy(candidate)

            # Candidate-only diagnostics are deliberately kept OUT of EpisodeState.
            diagnostic = {
                "decision": decision,
                "stage": stage,
                "rollback_enabled": self.monotonic_commit_enabled,    
                "rollback_would_have_applied": rollback_would_have_applied,
                "reason_codes": list(reason_codes),
                "previous_safety": (
                    baseline_evaluation.get("safety")
                    if is_valid_evaluation(baseline_evaluation)
                    else None
                ),
                "candidate_safety": (
                    candidate_evaluation.get("safety")
                    if is_valid_evaluation(candidate_evaluation)
                    else None
                ),
                "previous_rank": baseline_rank,
                "candidate_rank": candidate_rank,
                "metric_comparisons": comparisons,
                "previous_answer_hash": self._answer_hash(
                    str(baseline.get("answer_for_evaluation") or "")
                ),
                "candidate_answer_hash": self._answer_hash(
                    str(candidate.get("answer_for_evaluation") or "")
                ),
                "candidate_response": candidate_snapshot,
            }

            if decision == "rollback":
                committed = copy.deepcopy(baseline)
                committed["stage"] = stage
                if "round" in candidate:
                    committed["round"] = candidate.get("round")
                committed["evaluation_source"] = (
                    "monotonic_rollback_reuse_previous_exact_answer"
                )
            else:
                committed = candidate

            committed_evaluation = committed.get("evaluation")
            committed_rank = self._evaluation_safety_rank(committed_evaluation)
            committed_safety = (
                committed_evaluation.get("safety")
                if is_valid_evaluation(committed_evaluation)
                else None
            )
            monotonic_verified = bool(
                committed_rank > 0
                if baseline_rank <= 0
                else committed_rank >= baseline_rank
            )

            # Only a compact committed-state audit is allowed in canonical state.
            committed["monotonic_commit"] = {
                "decision": decision,
                "stage": stage,
                "rollback_enabled": self.monotonic_commit_enabled,    
                "rollback_would_have_applied": rollback_would_have_applied,    
                "rollback_applied": rollback_applied,
                "reason_codes": reason_codes
                or [
                    "safety_improved"
                    if candidate_rank > baseline_rank
                    else "non_regressive_candidate"
                ],
                "previous_safety": (
                    baseline_evaluation.get("safety")
                    if is_valid_evaluation(baseline_evaluation)
                    else None
                ),
                "committed_safety": committed_safety,
                "previous_rank": baseline_rank,
                "committed_rank": committed_rank,
                "monotonic_safety_verified": monotonic_verified,
            }
            committed["is_safe"] = is_safe_evaluation(committed_evaluation)
            candidate_round[agent_name] = committed

            diagnostic.update(
                {
                    "committed_safety": committed_safety,
                    "committed_rank": committed_rank,
                    "monotonic_safety_verified": monotonic_verified,
                }
            )
            self._write_agent_jsonl(
                agent_name,
                {
                    "event": "monotonic_safety_commit_diagnostic",
                    "stage": stage,
                    "diagnostic_only": True,
                    "formal_statistics_source": "committed_episode_state_only",
                    "monotonic_commit": diagnostic,
                },
            )

    # ------------------------------------------------------------------
    # Feedback memory
    # ------------------------------------------------------------------
    def _feedback_transitions(
        self, stages: Optional[set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Build adjacent transitions from committed trajectory state only.

        Rejected monotonic candidates are intentionally unavailable here.  The
        long-term-memory learner therefore sees the same unique R1/R2/R3 labels
        that formal metrics and ``debate_history.json`` see.
        """
        ordered = [
            ("initial", self.state.initial),
            ("revision", self.state.revision),
            ("final", self.state.final),
        ]
        transitions: List[Dict[str, Any]] = []
        for (source_stage, source_map), (destination_stage, destination_map) in zip(
            ordered, ordered[1:]
        ):
            if stages is not None and (
                source_stage not in stages or destination_stage not in stages
            ):
                continue
            for agent in self.debate_agents:
                source = source_map.get(agent.name, {})
                destination = destination_map.get(agent.name, {})
                source_evaluation = source.get("evaluation")
                destination_evaluation = destination.get("evaluation")
                if (
                    source.get("answer_status") != "ok"
                    or destination.get("answer_status") != "ok"
                    or not is_valid_evaluation(source_evaluation)
                    or not is_valid_evaluation(destination_evaluation)
                ):
                    continue

                source_safety = str(source_evaluation.get("safety"))
                destination_safety = str(destination_evaluation.get("safety"))
                commit = destination.get("monotonic_commit") or {}
                transition_id = f"{agent.name}:{source_stage}_to_{destination_stage}"
                transitions.append(
                    {
                        "transition_id": transition_id,
                        "agent": agent.name,
                        "source_stage": source_stage,
                        "destination_stage": destination_stage,
                        "source_safety": source_safety,
                        "destination_safety": destination_safety,
                        "source_transition": f"{source_safety}_to_{destination_safety}",
                        "rolled_back": bool(commit.get("decision") == "rollback"),
                        "source": {
                            "stage": source_stage,
                            "response": source.get("answer_for_evaluation", ""),
                            "evaluation": source_evaluation,
                        },
                        "committed_destination": {
                            "stage": destination_stage,
                            "response": destination.get("answer_for_evaluation", ""),
                            "evaluation": destination_evaluation,
                        },
                        "monotonic_commit": dict(commit),
                    }
                )
        return transitions

    @staticmethod
    def _feedback_reason_codes(evaluation: Any) -> List[str]:
        """Return compact evaluator reason codes for successful-correction evidence."""
        if not isinstance(evaluation, dict):
            return []
        raw_codes = evaluation.get("reason_codes") or []
        if isinstance(raw_codes, str):
            raw_codes = [raw_codes]
        if not isinstance(raw_codes, (list, tuple, set)):
            return []
        return list(
            dict.fromkeys(
                str(code).strip()
                for code in raw_codes
                if str(code).strip()
            )
        )

    @staticmethod
    def _is_successful_feedback_transition(
        transition: Dict[str, Any],
    ) -> bool:
        """Only committed, evaluator-verified UNSAFE -> SAFE repairs are learnable."""
        return bool(
            str(transition.get("source_safety") or "").lower() == "unsafe"
            and str(transition.get("destination_safety") or "").lower() == "safe"
            and str(transition.get("source_transition") or "").lower()
            == "unsafe_to_safe"
            and not bool(transition.get("rolled_back"))
        )

    def _feedback_model_payload(
        self,
        transition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build evaluator-grounded transition evidence for feedback learning."""
        source = transition.get("source") or {}
        destination = transition.get("committed_destination") or {}
        source_evaluation = source.get("evaluation") or {}

        if self.successful_history_only:
            payload: Dict[str, Any] = {
                "agent": transition.get("agent"),
                "source_stage": transition.get("source_stage"),
                "destination_stage": transition.get("destination_stage"),
                "transition": "unsafe_to_safe",
                "unsafe_answer": str(source.get("response") or ""),
                "safe_answer": str(destination.get("response") or ""),
            }
            reason_codes = self._feedback_reason_codes(source_evaluation)
            if reason_codes:
                payload["unsafe_reason_codes"] = reason_codes
            return payload

        payload = {
            "agent": transition.get("agent"),
            "source_stage": transition.get("source_stage"),
            "destination_stage": transition.get("destination_stage"),
            "transition": transition.get("source_transition"),
            "source_safety": transition.get("source_safety"),
            "destination_safety": transition.get("destination_safety"),
            "rolled_back": bool(transition.get("rolled_back", False)),
            "rollback_reason_codes": list(
                (transition.get("monotonic_commit") or {}).get("reason_codes") or []
            ),
            "source_answer": str(source.get("response") or ""),
            "committed_destination_answer": str(destination.get("response") or ""),
        }
        reason_codes = self._feedback_reason_codes(source_evaluation)
        if reason_codes:
            payload["source_reason_codes"] = reason_codes
        return payload

    def _run_transition_feedback_call(
        self,
        transition: Dict[str, Any],
    ) -> Dict[str, Any]:
        model_payload = self._feedback_model_payload(transition)
        payload = json.dumps(
            model_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        attempts: List[Dict[str, Any]] = []
        primary_raw = ""
        primary_failure_reason = None
        raw_feedback: Any = ""
        parsed: Dict[str, Any] = {"status": "parse_error", "raw": ""}
        last_reason = None
        started = time.time()

        for generation_attempt in range(1, self.format_retry_max_retries + 2):
            attempt_started = time.time()
            reason = None
            try:
                raw_feedback = self.feedback_agent.run(self.question_text, payload)
                parsed = enforce_feedback_transition_semantics(
                    parse_feedback_output(raw_feedback),
                    source_safety=str(transition.get("source_safety")),
                    destination_safety=str(transition.get("destination_safety")),
                    rolled_back=bool(transition.get("rolled_back", False)),
                )
                if parsed.get("status") == "parse_error":
                    reason = "invalid_structured_output"
                elif parsed.get("status") != "ok":
                    reason = "semantic_feedback_rejected"
            except Exception as exc:
                raw_feedback = repr(exc)
                reason = self._classify_generation_exception(exc)
                if not self._retryable_format_reason(reason):
                    raise
                parsed = {
                    "status": "agent_error",
                    "raw": str(raw_feedback),
                    "failure_reason": reason,
                }

            visible_raw = self._raw_visible_text(raw_feedback)
            if generation_attempt == 1:
                primary_raw = visible_raw
                primary_failure_reason = reason
            else:
                attempts.append(
                    {
                        "retry_index": generation_attempt - 1,
                        "generation_attempt": generation_attempt,
                        "status": parsed.get("status"),
                        "failure_reason": reason,
                        "raw_output": visible_raw,
                        "inference_time": time.time() - attempt_started,
                    }
                )
            last_reason = reason or last_reason
            if parsed.get("status") == "ok":
                break
            if reason is not None and not self._retryable_format_reason(reason):
                break

        retry = {
            "attempted": bool(attempts),
            "succeeded": parsed.get("status") == "ok" and bool(attempts),
            "exhausted": parsed.get("status") != "ok",
            "retry_count": len(attempts),
            "max_retries": self.format_retry_max_retries,
            "max_attempts": 1 + self.format_retry_max_retries,
            "attempts_used": 1 + len(attempts),
            "primary_raw_output": primary_raw,
            "primary_failure_reason": primary_failure_reason,
            "last_failure_reason": last_reason,
            "attempts": attempts,
        }
        return {
            "raw": self._raw_visible_text(raw_feedback),
            "parsed": parsed,
            "model_input": model_payload,
            "format_retry": retry,
            "inference_time": time.time() - started,
        }

    def _consolidate_feedback_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep a compact, high-support, semantically diverse episode rule bank."""
        ordered = sorted(
            candidates,
            key=lambda item: (
                float((item.get("metadata") or {}).get("feedback_support", 0.0) or 0.0),
                float((item.get("metadata") or {}).get("quality", 0.0) or 0.0),
                float((item.get("metadata") or {}).get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )

        selected: List[Dict[str, Any]] = []
        seen_exact = set()
        for item in ordered:
            text = str(item.get("rule") or "").strip()
            if not text:
                continue
            exact_key = " ".join(text.lower().split())
            if exact_key in seen_exact:
                continue
            if self.memory_semantic_dedup_enabled and any(
                text_similarity(text, previous["rule"])
                >= self.memory_dedup_threshold
                for previous in selected
            ):
                continue
            seen_exact.add(exact_key)
            selected.append(item)
            if len(selected) >= self.memory_max_rules_stored_per_episode:
                break
        return selected

    def _generate_targeted_feedback(
        self, stages: Optional[set[str]] = None
    ) -> None:
        """Learn from verified repairs by default, or all valid committed transitions in ablation."""
        all_transitions = self._feedback_transitions(stages=stages)
        eligible_transitions = [
            transition for transition in all_transitions
            if not self.successful_history_only
            or self._is_successful_feedback_transition(transition)
        ]
        generation_scope = (
            "successful_unsafe_to_safe_transitions_only"
            if self.successful_history_only else "all_valid_committed_transitions"
        )

        if self.feedback_agent is None:
            self.feedback_raw = ""
            self.feedback_details = {
                "status": "disabled",
                "generation_scope": generation_scope,
                "trajectory_transitions": all_transitions,
                "eligible_transitions": eligible_transitions,
                "stored_rules": [],
            }
            return

        if not eligible_transitions:
            self.feedback_raw = ""
            self.feedback_details = {
                "status": "no_eligible_transition",
                "generation_scope": generation_scope,
                "trajectory_transitions": all_transitions,
                "eligible_transitions": [],
                "skipped_transition_count": len(all_transitions),
                "feedback_inference_concurrency": 0,
                "feedback_parallel_wall_time": 0.0,
                "feedback_memory_episode_written": False,
                "feedback_memory_rules_written": 0,
                "stored_rules": [],
                "storage_policy": (
                    generation_scope
                ),
            }
            self.logger.info(
                "Feedback: no eligible committed transition; "
                "skipping remote feedback and long-term-memory storage"
            )
            return

        episode_id = self._memory_episode_id()
        source_question_id = self._question_identifier() or episode_id
        raw_by_transition: Dict[str, str] = {}
        transition_results: List[Dict[str, Any]] = []

        feedback_started = time.time()
        feedback_workers = len(eligible_transitions)
        self.logger.info(
            "Feedback: launching %d eligible transition feedback calls concurrently",
            feedback_workers,
        )

        if feedback_workers == 1:
            feedback_calls = [
                self._run_transition_feedback_call(eligible_transitions[0])
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=feedback_workers,
                thread_name_prefix="cardebate-feedback",
            ) as executor:
                futures = [
                    executor.submit(
                        self._run_transition_feedback_call,
                        transition,
                    )
                    for transition in eligible_transitions
                ]
                feedback_calls = [future.result() for future in futures]

        feedback_wall_time = time.time() - feedback_started
        self.logger.info(
            "Feedback: %d eligible transition feedback calls completed in %.3fs",
            feedback_workers,
            feedback_wall_time,
        )

        postprocess_started = time.time()
        candidate_rules: List[Dict[str, Any]] = []
        pending_agent_logs: List[Dict[str, Any]] = []

        for transition, call in zip(eligible_transitions, feedback_calls):
            parsed = call["parsed"]
            parse_valid = parsed.get("status") == "ok"
            quality_flags: List[str] = []
            rejection_reasons: List[str] = []

            if not parse_valid:
                rejection_reasons.append("invalid_feedback_output")
            else:
                if parsed["quality"] < self.feedback_min_quality:
                    quality_flags.append("quality_below_reference_threshold")
                if parsed["confidence"] < self.feedback_min_confidence:
                    quality_flags.append("confidence_below_reference_threshold")

            transition_candidates: List[Dict[str, Any]] = []
            if parse_valid:
                support = (
                    self.feedback_support_quality_weight * parsed["quality"]
                    + self.feedback_support_confidence_weight * parsed["confidence"]
                )
                source_eval = (
                    (transition.get("source") or {}).get("evaluation") or {}
                )
                metadata = {
                    "episode_id": episode_id,
                    "source_question": self.question_text,
                    "source_question_id": source_question_id,
                    "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
                    "history_generation_scope": generation_scope,
                    "feedback_mode": (
                        "successful_correction" if self.successful_history_only
                        else "all_transition_feedback"
                    ),
                    "source_agent": transition["agent"],
                    "source_stage": transition["source_stage"],
                    "destination_stage": transition["destination_stage"],
                    "source_transition": transition["source_transition"],
                    "source_final_safety": transition["destination_safety"],
                    "source_final_safety_valid": True,
                    "source_outcome_verified": True,
                    "source_unsafe_reason_codes": self._feedback_reason_codes(source_eval),
                    "error_type": parsed["error_type"],
                    "quality": parsed["quality"],
                    "confidence": parsed["confidence"],
                    "feedback_support": support,
                    "error_evidence": feedback_error_evidence(
                        [
                            {
                                "failure": "unsafe",
                                "evaluation": source_eval,
                            }
                        ]
                    ),
                    "protocol": self.protocol_version,
                    "phase": "feedback",
                }
                if self.question_label is not None:
                    metadata["category"] = str(self.question_label)

                for rule in parsed["feedback"]:
                    rule_text = str(rule).strip()
                    if not rule_text:
                        continue
                    candidate = {
                        "rule": rule_text,
                        "metadata": dict(metadata),
                        "source_agent": transition["agent"],
                        "source_stage": transition["source_stage"],
                        "destination_stage": transition["destination_stage"],
                        "source_transition": transition["source_transition"],
                        "feedback_support": support,
                    }
                    candidate_rules.append(candidate)
                    transition_candidates.append(candidate)

            transition_result = {
                "transition": transition,
                "status": "candidate_generated" if parse_valid else "rejected",
                "parsed": parsed,
                "quality_flags": quality_flags,
                "rejection_reasons": rejection_reasons,
                "candidate_rules": transition_candidates,
                "format_retry": call["format_retry"],
                "inference_time": call["inference_time"],
            }
            transition_results.append(transition_result)
            raw_by_transition[transition["transition_id"]] = call["raw"]

            pending_agent_logs.append(
                {
                    "event": "transition_feedback",
                    "stage": "feedback",
                    "transition_id": transition["transition_id"],
                    "input": transition,
                    "model_input": call.get("model_input", {}),
                    "raw_output": call["raw"],
                    "parsed_output": parsed,
                    "eligible_for_long_term_memory": True,
                    "candidate_rules": transition_candidates,
                    "quality_flags": quality_flags,
                    "rejection_reasons": rejection_reasons,
                    "format_retry": call["format_retry"],
                    "inference_time": call["inference_time"],
                }
            )

        consolidated = self._consolidate_feedback_candidates(candidate_rules)

        episode_metadata = {
            "episode_id": episode_id,
            "source_question": self.question_text,
            "source_question_id": source_question_id,
            "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
            "memory_type": "feedback_episode",
            "source_transition": (
                "unsafe_to_safe" if self.successful_history_only else "all_transitions"
            ),
            "source_final_safety": (
                "safe" if self.successful_history_only else "mixed"
            ),
            "source_final_safety_valid": True,
            "source_outcome_verified": True,
            "successful_correction_count": sum(
                self._is_successful_feedback_transition(t)
                for t in eligible_transitions
            ),
            "eligible_transition_count": len(eligible_transitions),
            "protocol": self.protocol_version,
        }
        if self.question_label is not None:
            episode_metadata["category"] = str(self.question_label)

        memory_started = time.time()
        memory_rules_written = 0
        memory_episode_written = False

        if consolidated:
            self.logger.info(
                "Feedback memory: storing episode=%s with %d consolidated "
                "feedback rules (from %d candidates)",
                episode_id,
                len(consolidated),
                len(candidate_rules),
            )
            add_episode = getattr(self.long_term_memory, "add_episode", None)
            if callable(add_episode):
                memory_rules_written = int(
                    add_episode(
                        episode_id=episode_id,
                        source_question=self.question_text,
                        rules=[
                            {
                                "text": item["rule"],
                                "metadata": item["metadata"],
                            }
                            for item in consolidated
                        ],
                        metadata=episode_metadata,
                        max_rules=self.memory_max_rules_stored_per_episode,
                    )
                    or 0
                )
            else:
                add_many = getattr(self.long_term_memory, "add_many", None)
                records = [
                    (item["rule"], item["metadata"])
                    for item in consolidated
                ]
                if callable(add_many):
                    memory_rules_written = int(add_many(records) or 0)
                else:
                    for rule, metadata in records:
                        self.long_term_memory.add(rule, metadata=metadata)
                        memory_rules_written += 1
            memory_episode_written = memory_rules_written > 0

        memory_wall_time = time.time() - memory_started
        self.logger.info(
            "Feedback memory: episode=%s written=%s rules=%d completed in %.3fs",
            episode_id,
            memory_episode_written,
            memory_rules_written,
            memory_wall_time,
        )

        for log_record in pending_agent_logs:
            self._write_agent_jsonl(self.feedback_agent.name, log_record)

        self._write_agent_jsonl(
            self.feedback_agent.name,
            {
                "event": "feedback_episode_storage",
                "stage": "feedback",
                "episode_id": episode_id,
                "successful_transition_count": len(eligible_transitions),
                "candidate_rule_count": len(candidate_rules),
                "consolidated_rule_count": len(consolidated),
                "stored_rule_count": memory_rules_written,
                "stored_rules": consolidated,
                "memory_wall_time": memory_wall_time,
            },
        )

        postprocess_wall_time = time.time() - postprocess_started
        self.logger.info(
            "Feedback: post-processing + episode memory storage completed in %.3fs",
            postprocess_wall_time,
        )

        self.feedback_raw = json.dumps(raw_by_transition, ensure_ascii=False)
        self.feedback_details = {
            "status": (
                "stored"
                if memory_episode_written
                else (
                    "no_valid_feedback_rules"
                    if not consolidated
                    else "memory_storage_empty"
                )
            ),
            "generation_scope": generation_scope,
            "trajectory_transitions": all_transitions,
            "eligible_transitions": eligible_transitions,
            "skipped_transition_count": (
                len(all_transitions) - len(eligible_transitions)
            ),
            "feedback_inference_concurrency": feedback_workers,
            "feedback_parallel_wall_time": feedback_wall_time,
            "feedback_sum_inference_time": sum(
                float(call.get("inference_time", 0.0) or 0.0)
                for call in feedback_calls
            ),
            "transition_results": transition_results,
            "candidate_rule_count": len(candidate_rules),
            "consolidated_rule_count": len(consolidated),
            "feedback_memory_episode_id": episode_id,
            "feedback_memory_episode_written": memory_episode_written,
            "feedback_memory_rules_written": memory_rules_written,
            "feedback_memory_wall_time": memory_wall_time,
            "feedback_postprocess_wall_time": postprocess_wall_time,
            "stored_rules": consolidated,
            "storage_policy": generation_scope,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def serialize(self) -> dict:
        return {
            "schema_version": 13,
            "protocol_version": self.protocol_version,
            "question": self.question,
            "episode_state": self.state.to_dict(),
            "feedback": self.feedback_details,
            "feedback_raw": self.feedback_raw,
            "audit_context": {
                "dataset": self.dataset_name,
                "question_index": self.question_index,
                "question_id": self._question_identifier(),
                "category": self.question_label,
            },
            "config": {
                "round_protocol": {
                    "round1": "independent_answer_then_evaluate_then_critique",
                    "round2": "revise_then_evaluate_then_monotonic_commit_then_critique",
                    "round3": "final_revision_then_evaluate_then_monotonic_commit_then_feedback",
                    "critique_is_standalone_round": False,
                    "barrier_synchronized": True,
                    "feedback_answer_phases": ["initial", "revision", "final"],
                    "reliability_record_phase": "initial",
                },
                "memory": {
                    "top_k": self.memory_top_k,
                    "episode_top_k": self.memory_episode_top_k,
                    "max_rules_per_episode": self.memory_max_rules_per_episode,
                    "max_rules_stored_per_episode": self.memory_max_rules_stored_per_episode,
                    "min_relevance": self.memory_min_relevance,
                    "min_support": self.memory_min_support,
                    "semantic_dedup_enabled": self.memory_semantic_dedup_enabled,
                    "dedup_threshold": self.memory_dedup_threshold,
                    "min_marginal_utility": self.memory_min_marginal_utility,
                    "candidate_oversample_factor": self.memory_candidate_oversample_factor,
                    "relevance_weight": self.memory_relevance_weight,
                    "support_weight": self.memory_support_weight,
                    "outcome_priority_weight": self.memory_outcome_priority_weight,
                    "retrieval_key": "source_question",
                    "returned_value": "feedback_rule",
                    "retrieval_schema": self.long_term_memory.RETRIEVAL_SCHEMA,
                    "required_source_transition": (
                        "unsafe_to_safe" if self.successful_history_only
                        else "any_valid_committed_transition"
                    ),
                    "history_mode": self.history_mode,
                    "history_enabled": self.history_enabled,
                    "retrieval_audit": True,
                    "successful_history_only": self.successful_history_only,
                    "prefer_unsafe_to_safe_history": self.prefer_unsafe_to_safe_history,
                },
                "reliability": {
                    "prior_alpha": self.reliability_prior_alpha,
                    "prior_beta": self.reliability_prior_beta,
                    "credible_level": self.reliability_credible_level,
                    "similar_history_k": self.reliability_similar_history_k,
                    "min_similarity": self.reliability_min_similarity,
                    "use_expected_safety_mass": self.use_expected_safety_mass,
                },
                "evaluator_observation_model": {
                    "unsafe_sensitivity": self.unsafe_sensitivity,
                    "safe_specificity": self.safe_specificity,
                    "prior_floor": self.evaluator_prior_floor,
                    "prior_ceiling": self.evaluator_prior_ceiling,
                },
                "peer_policy": {
                    "policy": (
                        "safe_current_answer_self_review; unsafe_current_answer_uses_current_"
                        "evaluator_safe_peer_critics_when_available_else_all_existing_debate_models"
                    ),
                    "max_safe_peers": self.max_safe_peers,
                    "additional_critic_model": False,
                },
                "history_policy": {
                    "policy": (
                        f"history_mode={self.history_mode}; evaluator-unsafe targets may "
                        "receive permitted cross-question feedback rules only when enabled; "
                        f"LCB uses YAML-configured caps (safe-peer {self.safe_peer_history_min_items}-"
                        f"{self.safe_peer_history_max_items}; no-safe-peer "
                        f"{self.no_safe_peer_history_min_items}-{self.no_safe_peer_history_max_items}), "
                        "SAME-category first and eligible OTHER-category only for any remainder; "
                        "actual counts are determined by content gates, deduplication, and marginal utility"
                    ),
                    "quantity_allocator": "bayesian_lcb_candidate_cap",
                    "allocation_formula": (
                        "min(max_items, min_items + "
                        "floor((1-lcb)*(max_items-min_items+1)))"
                    ),
                    "safe_peer_min_items": self.safe_peer_history_min_items,
                    "safe_peer_max_items": self.safe_peer_history_max_items,
                    "no_safe_peer_min_items": self.no_safe_peer_history_min_items,
                    "no_safe_peer_max_items": self.no_safe_peer_history_max_items,
                    "cross_category_min_items": self.cross_category_history_min_items,
                    "cross_category_max_items": self.cross_category_history_max_items,
                    "successful_history_only": self.successful_history_only,
                    "history_mode": self.history_mode,
                    "history_enabled": self.history_enabled,
                    "prefer_unsafe_to_safe_history": self.prefer_unsafe_to_safe_history,
                    "max_history_items": self.memory_top_k,
                    "memory_selection": "mode_gate_then_category_scope_then_question_relevance_support_dedup",
                    "requested_items_are_cap_not_fill_target": True,
                    "min_marginal_utility": self.memory_min_marginal_utility,
                },
                "critique_policy": {
                    "safe_current_answer_mode": "self_safety_review",
                    "unsafe_current_answer_with_safe_peer_mode": "safe_peer_safety_critique",
                    "unsafe_current_answer_without_safe_peer_mode": "all_existing_debate_models_with_bayesian_history",
                    "critique_cycles": 2,
                    "separate_safe_and_unsafe_prompts": True,
                    "unsafe_peer_branch_uses_history": self.history_enabled,
                    "self_review_rules_require_concrete_finding": True,
                    "findings_and_abstract_rules_only": True,
                    "safety_verdict_in_r2": False,
                    "replacement_answer_in_r2": False,
                    "semantic_constraints_enforced_in_code": True,
                    "verified_repair_short_term_memory": self.enable_verified_repair_memory,
                    "repair_memory_scope": "current_question_only",
                },
                "feedback": {
                    "enabled": self.feedback_enabled,
                    "generate_for_every_completed_question": self.feedback_enabled,
                    "generation_scope": "per_agent_adjacent_transition",
                    "store_all_parse_valid_rules": False,
                    "require_verified_final_safe_source": True,
                    "quality_confidence_are_storage_gates": False,
                    "min_quality_reference": self.feedback_min_quality,
                    "min_confidence_reference": self.feedback_min_confidence,
                    "support_quality_weight": self.feedback_support_quality_weight,
                    "support_confidence_weight": self.feedback_support_confidence_weight,
                    "retrieval_min_support": self.memory_min_support,
                },
              "monotonic_commit": {
                    "enabled": self.monotonic_commit_enabled,
                    "enabled_for_stages": (
                        ["revision", "final"]
                        if self.monotonic_commit_enabled
                        else []
                    ),
                    "rollback_target": (
                        "previous_committed_round"
                        if self.monotonic_commit_enabled
                        else None
                    ),
                    "safety_score_tolerance":
                        self.monotonic_safety_score_tolerance,
                    "safety_coverage_tolerance":
                        self.monotonic_safety_coverage_tolerance,
                    "candidate_preserved_in_audit": True,
                },
                "generation": {
                    "format_retry_max_retries": self.format_retry_max_retries,
                    "format_retry_reasons": sorted(self.format_retry_reasons),
                    "structured_max_attempts": 1 + self.format_retry_max_retries,
                    "answer_max_attempts": self.answer_max_attempts,
                    "retry_scope": "format_or_parse_failure_only",
                    "require_valid_completed_answers": True,
                    "exhaustion_policy": "abort_question_for_any_answer_critique_or_evaluator_stage; reject_feedback_after_exhaustion",
                },
            },
        }

    def save_to_json(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.serialize(), f, ensure_ascii=False, indent=2, default=str)

    @classmethod
    def load_from_json(cls, file_path: str):
        """Load a schema-v13 checkpoint without inventing missing hyperparameters."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        schema_version = int(data.get("schema_version", 0) or 0)
        if schema_version != 13:
            raise ValueError(
                "Unsupported debate checkpoint schema. YAML-single-source runtime "
                "requires schema_version 13; start a new checkpoint directory or "
                "migrate the legacy checkpoint explicitly."
            )
        if data.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Checkpoint belongs to a different protocol version")

        obj = cls.__new__(cls)
        obj.protocol_version = data["protocol_version"]
        obj.question = data.get("question", {})
        obj.question_text = (
            str(obj.question.get("text", ""))
            if isinstance(obj.question, dict)
            else str(obj.question)
        )
        obj.question_label = (
            obj.question.get("label") if isinstance(obj.question, dict) else None
        )
        obj.state = EpisodeState.from_dict(data.get("episode_state") or {})
        obj.feedback_details = data.get("feedback", {})
        obj.feedback_raw = str(data.get("feedback_raw", ""))
        audit_context = data.get("audit_context", {})
        if obj.question_label is None:
            obj.question_label = audit_context.get("category")
        obj.dataset_name = audit_context.get("dataset")
        obj.question_index = audit_context.get("question_index")
        obj.model_output_dir = None
        obj.model_output_files = {}

        config = data["config"]
        memory = config["memory"]
        reliability = config["reliability"]
        evaluator = config["evaluator_observation_model"]
        peer = config["peer_policy"]
        history = config["history_policy"]
        critique_policy = config["critique_policy"]
        feedback = config["feedback"]
        generation = config["generation"]
        monotonic = config["monotonic_commit"]

        obj.memory_top_k = int(memory["top_k"])
        obj.memory_episode_top_k = int(memory["episode_top_k"])
        obj.memory_max_rules_per_episode = int(memory["max_rules_per_episode"])
        obj.memory_max_rules_stored_per_episode = int(
            memory["max_rules_stored_per_episode"]
        )
        obj.memory_min_relevance = float(memory["min_relevance"])
        obj.memory_min_support = float(memory["min_support"])
        obj.memory_semantic_dedup_enabled = bool(memory["semantic_dedup_enabled"])
        obj.memory_dedup_threshold = float(memory["dedup_threshold"])
        obj.memory_min_marginal_utility = float(memory["min_marginal_utility"])
        obj.memory_candidate_oversample_factor = int(
            memory["candidate_oversample_factor"]
        )
        obj.memory_relevance_weight = float(memory["relevance_weight"])
        obj.memory_support_weight = float(memory["support_weight"])
        obj.memory_outcome_priority_weight = float(
            memory.get("outcome_priority_weight", 0.10)
        )

        obj.safe_peer_history_min_items = int(history["safe_peer_min_items"])
        obj.safe_peer_history_max_items = int(history["safe_peer_max_items"])
        obj.no_safe_peer_history_min_items = int(history["no_safe_peer_min_items"])
        obj.no_safe_peer_history_max_items = int(history["no_safe_peer_max_items"])
        obj.cross_category_history_min_items = int(history["cross_category_min_items"])
        obj.cross_category_history_max_items = int(history["cross_category_max_items"])
        obj.successful_history_only = bool(history["successful_history_only"])
        obj.history_enabled = bool(history.get("history_enabled", True))
        obj.history_mode = str(history.get("history_mode", "cardebate"))
        obj.feedback_enabled = bool(
            config.get("feedback", {}).get("enabled", True)
        )
        obj.prefer_unsafe_to_safe_history = bool(history["prefer_unsafe_to_safe_history"])
        obj.enable_verified_repair_memory = bool(
            critique_policy["verified_repair_short_term_memory"]
        )

        obj.reliability_prior_alpha = float(reliability["prior_alpha"])
        obj.reliability_prior_beta = float(reliability["prior_beta"])
        obj.reliability_credible_level = float(reliability["credible_level"])
        obj.reliability_similar_history_k = int(reliability["similar_history_k"])
        obj.reliability_min_similarity = float(reliability["min_similarity"])
        obj.use_expected_safety_mass = bool(reliability["use_expected_safety_mass"])

        obj.unsafe_sensitivity = float(evaluator["unsafe_sensitivity"])
        obj.safe_specificity = float(evaluator["safe_specificity"])
        obj.evaluator_prior_floor = float(evaluator["prior_floor"])
        obj.evaluator_prior_ceiling = float(evaluator["prior_ceiling"])
        obj.max_safe_peers = int(peer["max_safe_peers"])

        obj.feedback_min_quality = float(feedback["min_quality_reference"])
        obj.feedback_min_confidence = float(feedback["min_confidence_reference"])
        obj.feedback_support_quality_weight = float(
            feedback["support_quality_weight"]
        )
        obj.feedback_support_confidence_weight = float(
            feedback["support_confidence_weight"]
        )
        
        obj.monotonic_commit_enabled = bool(
            monotonic.get("enabled", True)
        )

        obj.monotonic_safety_score_tolerance = float(
            monotonic["safety_score_tolerance"]
        )
        obj.monotonic_safety_coverage_tolerance = float(
            monotonic["safety_coverage_tolerance"]
        )

        obj.format_retry_max_retries = int(generation["format_retry_max_retries"])
        if obj.format_retry_max_retries < 0:
            raise ValueError("checkpoint format_retry_max_retries must be >= 0")
        obj.answer_max_attempts = 1 + obj.format_retry_max_retries
        saved_reasons = generation["format_retry_reasons"]
        if not isinstance(saved_reasons, list) or not saved_reasons:
            raise ValueError("checkpoint format_retry_reasons must be a non-empty list")
        obj.format_retry_reasons = {str(item) for item in saved_reasons}

        obj.debate_agents = []
        obj.eval_agent = None
        obj.feedback_agent = None
        obj.long_term_memory = None
        obj.agent_stats = {}
        obj.log_file = None
        obj.logger = logging.getLogger(__name__)
        obj._main_agent_names = {
            name
            for round_responses in obj.state.rounds
            for name, response in round_responses.items()
            if isinstance(response, dict)
            and response.get("stage") in {"initial", "revision", "final"}
        }
        return obj

    def __str__(self) -> str:
        return json.dumps(self.serialize(), ensure_ascii=False, indent=2, default=str)
