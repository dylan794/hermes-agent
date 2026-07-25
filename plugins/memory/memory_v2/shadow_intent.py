"""Disabled-by-default memory-need routing for shadow retrieval experiments.

The production router answers which memory route to use.  This module answers
the earlier question: whether long-term memory should be consulted at all.
It is intentionally conservative and has no mutation or tool authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|k|thanks?|thank you|great|nice|cool|got it|sounds good)[.! ]*$",
    re.IGNORECASE,
)
_SIMPLE_ARITHMETIC_RE = re.compile(
    r"^\s*(?:what(?:'s| is)\s+)?[-+*/().\d\s]+\??\s*$",
    re.IGNORECASE,
)
_CURRENT_CONTEXT_RE = re.compile(
    r"\b(?:this|that|those|these|above|after your message|popped up|the error|"
    r"those steps|these steps|what you (?:just )?(?:said|described))\b",
    re.IGNORECASE,
)
_HISTORY_RE = re.compile(
    r"\b(?:previous(?:ly)?|earlier|last (?:time|week|month|year)|"
    r"history|historical|used to|what did we|why did we)\b",
    re.IGNORECASE,
)
_CURRENT_STATE_RE = re.compile(
    r"\b(?:current(?:ly)?|latest|status|where are we|where we are|up to date)\b",
    re.IGNORECASE,
)
_EXPLICIT_CURRENT_TARGET_RE = re.compile(
    r"\b(?:current(?:ly)?|latest|where are we|where we are|up to date)\b",
    re.IGNORECASE,
)
_EXPLICIT_MEMORY_RE = re.compile(
    r"\b(?:remember|recall|from memory|past conversation|last time|"
    r"where (?:did we|were we) leave off)\b",
    re.IGNORECASE,
)
_RESUME_RE = re.compile(
    r"\b(?:resume|continue|pick up|where we left off|restart the .* work)\b",
    re.IGNORECASE,
)
_GENERIC_CONTINUITY_RE = re.compile(
    r"\b(?:what(?:'s| is) next|next steps?|what should we do|where are we)\b",
    re.IGNORECASE,
)
_MAX_GAP_DAYS = 36_525.0


@dataclass(frozen=True)
class MemoryNeedDecision:
    """Read-only routing decision for a shadow retrieval run."""

    decision: str
    reason: str
    confidence: float
    temporal_mode: str
    search_limit: int
    allow_unknown_workstream: bool
    mutation_authority: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "confidence": self.confidence,
            "temporal_mode": self.temporal_mode,
            "search_limit": self.search_limit,
            "allow_unknown_workstream": self.allow_unknown_workstream,
            "mutation_authority": self.mutation_authority,
        }


class ShadowMemoryNeedRouter:
    """Decide whether a bounded shadow search is warranted.

    ``uncertain`` is not permission for broad raw prefetch.  It permits only a
    small, profile-scoped candidate search whose downstream ranker can abstain.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def route(
        self,
        query: str,
        context: Mapping[str, Any] | None = None,
    ) -> MemoryNeedDecision:
        if not self.enabled:
            return self._none("shadow_router_disabled")
        normalized = " ".join(str(query or "").split())
        parsed = self._context(context)
        if not normalized or parsed is None:
            return self._none("invalid_context")

        has_current_context = parsed["has_current_context"]
        gap_days = parsed["gap_days"]
        workstreams = parsed["workstream_ids"]
        history_signal = bool(_HISTORY_RE.search(normalized))
        current_target = bool(_EXPLICIT_CURRENT_TARGET_RE.search(normalized))
        temporal_mode = (
            "history" if history_signal and not current_target else "current"
        )

        if _ACKNOWLEDGEMENT_RE.match(normalized) or _SIMPLE_ARITHMETIC_RE.match(
            normalized
        ):
            return self._none("memory_not_useful", temporal_mode=temporal_mode)

        if (
            has_current_context
            and gap_days <= 1.0
            and _CURRENT_CONTEXT_RE.search(normalized)
            and not _EXPLICIT_MEMORY_RE.search(normalized)
            and not _HISTORY_RE.search(normalized)
        ):
            return self._none(
                "current_context_sufficient",
                temporal_mode=temporal_mode,
            )

        if (
            _EXPLICIT_MEMORY_RE.search(normalized)
            or _HISTORY_RE.search(normalized)
            or (
                _RESUME_RE.search(normalized)
                and (gap_days >= 1.0 or bool(workstreams))
                and not has_current_context
            )
            or (_CURRENT_STATE_RE.search(normalized) and not has_current_context)
        ):
            return MemoryNeedDecision(
                decision="needed",
                reason="explicit_long_term_need",
                confidence=0.9,
                temporal_mode=temporal_mode,
                search_limit=20,
                allow_unknown_workstream=not bool(workstreams),
            )

        if (
            _GENERIC_CONTINUITY_RE.search(normalized)
            and not has_current_context
            and gap_days >= 1.0
        ):
            return MemoryNeedDecision(
                decision="uncertain",
                reason="ambiguous_continuity_query",
                confidence=0.4,
                temporal_mode=temporal_mode,
                search_limit=5,
                allow_unknown_workstream=True,
            )

        return self._none("no_long_term_memory_signal", temporal_mode=temporal_mode)

    @staticmethod
    def _context(
        value: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            return None
        has_current_context = value.get("has_current_context", False)
        if not isinstance(has_current_context, bool):
            return None
        gap_days = value.get("gap_days", 0.0)
        if isinstance(gap_days, bool):
            return None
        try:
            gap = float(gap_days)
        except (OverflowError, TypeError, ValueError):
            return None
        if not math.isfinite(gap) or gap < 0.0 or gap > _MAX_GAP_DAYS:
            return None
        raw_workstreams = value.get("workstream_ids") or []
        if not isinstance(raw_workstreams, (list, tuple)):
            return None
        workstreams = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in raw_workstreams
                if str(item).strip()
            )
        )
        return {
            "has_current_context": has_current_context,
            "gap_days": gap,
            "workstream_ids": workstreams,
        }

    @staticmethod
    def _none(
        reason: str,
        *,
        temporal_mode: str = "current",
    ) -> MemoryNeedDecision:
        return MemoryNeedDecision(
            decision="none",
            reason=reason,
            confidence=1.0,
            temporal_mode=temporal_mode,
            search_limit=0,
            allow_unknown_workstream=False,
        )
