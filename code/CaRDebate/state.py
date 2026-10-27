"""Structured short-term state for one CaRDebate question.

The protocol needs short-term memory, but it is represented as explicit Python
state rather than a prompt-memory object.  Models never receive this object as a
whole.  The orchestrator selects only the fields that are appropriate for the
current stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


ResponseMap = Dict[str, Dict[str, Any]]


@dataclass
class EpisodeState:
    """All transient state produced while solving one question.

    `initial`, `critiques`, and `final` are the three protocol stages.
    `peer_routing` and `experience_selection` are policy-side audit records.
    `reliability_snapshot` is frozen before R1 so current observations cannot
    influence the current question's Bayesian reliability.
    """

    initial: ResponseMap = field(default_factory=dict)
    critiques: ResponseMap = field(default_factory=dict)
    final: ResponseMap = field(default_factory=dict)
    peer_routing: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    experience_selection: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    reliability_snapshot: Dict[str, Any] = field(default_factory=dict)
    question_embedding: Optional[List[float]] = None

    @property
    def rounds(self) -> list[ResponseMap]:
        """Return populated stage maps in protocol order."""
        result: list[ResponseMap] = []
        if self.initial:
            result.append(self.initial)
        if self.critiques:
            result.append(self.critiques)
        if self.final:
            result.append(self.final)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial": self.initial,
            "critiques": self.critiques,
            "final": self.final,
            "peer_routing": self.peer_routing,
            "experience_selection": self.experience_selection,
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
            final=dict(data.get("final") or {}),
            peer_routing=dict(data.get("peer_routing") or {}),
            experience_selection=dict(data.get("experience_selection") or {}),
            reliability_snapshot=dict(data.get("reliability_snapshot") or {}),
            question_embedding=(
                list(data["question_embedding"])
                if isinstance(data.get("question_embedding"), list)
                else None
            ),
        )
