"""Offline source-grounded extraction for Memory v2.

The extractor is intentionally deterministic and conservative. It scans archived
raw turn evidence after/near session end and creates *pending* CandidateMemory
records only. It never writes durable MemoryItem/ProjectCard/open-loop records;
manual review or a later consolidation pass owns promotion.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .index import MemoryV2Index
from .redaction import contains_sensitive_text, redact_text
from .schemas import CandidateMemory, GateDecision, MemoryStatus, MemoryType, ValidationError, normalize_project_id
from .store import MemoryV2Store


@dataclass
class ExtractionReport:
    """Summary of one offline extraction pass."""

    considered_events: int = 0
    created: int = 0
    merged: int = 0
    skipped: int = 0
    created_ids: List[str] = field(default_factory=list)
    merged_ids: List[str] = field(default_factory=list)
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "considered_events": self.considered_events,
            "created": self.created,
            "merged": self.merged,
            "skipped": self.skipped,
            "created_ids": list(self.created_ids),
            "merged_ids": list(self.merged_ids),
            "skipped_reasons": dict(self.skipped_reasons),
        }


@dataclass(frozen=True)
class _ExtractedCandidate:
    type: str
    claim: str
    proposed_destination: str
    promotion_reason: str
    confidence: float
    importance: float
    source_refs: List[str]


class OfflineSessionExtractor:
    """Extract high-precision pending candidates from raw session evidence."""

    version = "offline_extraction:v1"

    _EXPLICIT_MEMORY_RE = re.compile(
        r"^\s*(?:please\s+)?(?:remember|don't\s+forget|do\s+not\s+forget)\s+(?:that\s+)?",
        re.IGNORECASE,
    )
    _SENDER_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")
    _PROJECT_FIELD_RE = re.compile(
        r"^\s*(?P<project>[A-Z][A-Za-z0-9 _.-]{1,80}?)\s+"
        r"(?P<kind>decision|next action|open question|current state|goal|status)\s*:\s*(?P<value>.+?)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    _ENV_RE = re.compile(r"^\s*(?:Hermes\s+)?environment fact\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.DOTALL)
    _PREFERENCE_PATTERNS = (
        re.compile(r"^\s*I\s+prefer\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*Actually,?\s+I\s+prefer\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*My\s+preference\s+is\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.DOTALL),
    )
    _OPEN_LOOP_PATTERNS = (
        re.compile(r"^\s*(?:we|you|i)\s+need\s+to\s+(?P<value>follow\s+up\s+.+?)\s*$", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*follow\s+up\s+on\s+(?P<value>.+?)\s*$", re.IGNORECASE | re.DOTALL),
    )
    _SKILL_RE = re.compile(
        r"\b(?:save|saved|remember|capture)\b.+\b(?:as|into)\s+a\s+skill\b|\bshould\s+be\s+saved\s+as\s+a\s+skill\b",
        re.IGNORECASE | re.DOTALL,
    )
    _EPHEMERAL_RE = re.compile(
        r"\b(today|tonight|tomorrow|for now|right now|currently|temporary|temporarily|this morning|this afternoon)\b",
        re.IGNORECASE,
    )
    _SCOPED_CURRENT_RE = re.compile(
        r"\b(this answer|this reply|this response|this message|this conversation|this thread|for this one|in this case)\b",
        re.IGNORECASE,
    )
    _INSTRUCTION_BAIT_RE = re.compile(
        r"\b(system:|developer:|ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions|tool_call|function_call|memory_v2_promote|candidate_id|reveal\s+secrets?|call\s+tool)\b",
        re.IGNORECASE,
    )

    def extract(
        self,
        store: MemoryV2Store,
        index: MemoryV2Index,
        *,
        session_id: str = "",
        recent_raw_limit: Optional[int] = None,
    ) -> ExtractionReport:
        """Extract pending candidates from raw events and index them.

        ``session_id`` filters raw turn evidence to one session when supplied.
        The method is idempotent for identical normalized claims: existing
        candidates are reused and pending candidates get merged source refs.
        """
        report = ExtractionReport()
        for event in self._events(store, session_id=session_id, limit=recent_raw_limit):
            report.considered_events += 1
            event_id = str(event.get("id") or "").strip()
            if str(event.get("type") or "") != "turn" or not str(event.get("user_content") or "").strip():
                report.skip("not_user_turn_evidence")
                continue
            text = self._event_user_text(event)
            if not event_id or not text:
                report.skip("empty_event")
                continue
            if self._EXPLICIT_MEMORY_RE.match(text):
                report.skip("explicit_memory_handled_online")
                continue
            extracted = self._extract_from_text(text, source_id=event_id)
            if extracted is None:
                report.skip("no_durable_candidate")
                continue
            if not self._source_exists(store, event_id):
                report.skip("missing_source_ref")
                continue
            candidate = self._candidate_from_extracted(extracted)
            result = self._upsert_candidate(store, index, candidate)
            if result == "created":
                report.created += 1
                report.created_ids.append(candidate.id)
            elif result == "merged":
                report.merged += 1
                report.merged_ids.append(candidate.id)
            else:
                report.skip(result)
        return report

    def _events(self, store: MemoryV2Store, *, session_id: str, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
        safe_limit = 50 if limit is None else int(limit)
        try:
            return store.search_raw_events("", session_id=str(session_id or ""), limit=safe_limit)
        except ValidationError:
            # Hot extraction paths fail closed when the derived raw index is
            # unavailable instead of falling back to a hidden full-archive scan.
            return []

    def _extract_from_text(self, text: str, *, source_id: str) -> Optional[_ExtractedCandidate]:
        clean = self._clean_text(text)
        if self._should_skip_text(clean):
            return None
        if match := self._PROJECT_FIELD_RE.match(clean):
            return self._project_candidate(match, source_id=source_id)
        if match := self._ENV_RE.match(clean):
            value = self._finish_sentence(match.group("value"))
            return _ExtractedCandidate(
                type=MemoryType.ENVIRONMENT.value,
                claim=f"Hermes environment fact: {value}",
                proposed_destination="semantic/items",
                promotion_reason=f"{self.version} environment_fact: extracted from raw session evidence; pending review required.",
                confidence=0.74,
                importance=0.62,
                source_refs=[source_id],
            )
        for pattern in self._PREFERENCE_PATTERNS:
            if match := pattern.match(clean):
                value = self._finish_sentence(match.group("value"))
                return _ExtractedCandidate(
                    type=MemoryType.PREFERENCE.value,
                    claim=f"User prefers {value}",
                    proposed_destination="semantic/items",
                    promotion_reason=f"{self.version} preference: extracted from first-person user statement; pending review required.",
                    confidence=0.78,
                    importance=0.68,
                    source_refs=[source_id],
                )
        if self._SKILL_RE.search(clean):
            return _ExtractedCandidate(
                type=MemoryType.PROCEDURE_REF.value,
                claim=self._finish_sentence(clean),
                proposed_destination="skills",
                promotion_reason=f"{self.version} skill_candidate: user indicated a reusable workflow may belong in procedural memory; pending skill-authoring review required.",
                confidence=0.68,
                importance=0.72,
                source_refs=[source_id],
            )
        for pattern in self._OPEN_LOOP_PATTERNS:
            if match := pattern.match(clean):
                value = self._finish_sentence(match.group("value"))
                return _ExtractedCandidate(
                    type=MemoryType.PROJECT_STATE.value,
                    claim=f"Open loop: {value}",
                    proposed_destination="working/open_loops.yaml",
                    promotion_reason=f"{self.version} open_loop: extracted from user-stated follow-up; pending review required.",
                    confidence=0.66,
                    importance=0.64,
                    source_refs=[source_id],
                )
        return None

    def _project_candidate(self, match: re.Match[str], *, source_id: str) -> _ExtractedCandidate:
        project = self._finish_sentence(match.group("project")).rstrip(".")
        if project.lower().startswith("project "):
            project = project[len("project ") :].strip()
        kind = str(match.group("kind") or "").lower().replace(" ", "_")
        value = self._finish_sentence(match.group("value"))
        project_id = normalize_project_id(project)
        slug = project_id.split(":", 1)[1]
        label = kind.replace("_", " ")
        return _ExtractedCandidate(
            type=MemoryType.PROJECT_STATE.value,
            claim=f"Project {project} {label}: {value}",
            proposed_destination=f"semantic/projects/{slug}.yaml",
            promotion_reason=f"{self.version} project_{kind}: extracted from labeled user project update; pending review required.",
            confidence=0.76,
            importance=0.7 if kind in {"decision", "next_action"} else 0.62,
            source_refs=[source_id],
        )

    def _candidate_from_extracted(self, extracted: _ExtractedCandidate) -> CandidateMemory:
        normalized = self._dedupe_key_parts(extracted.type, extracted.proposed_destination, extracted.claim)
        digest = hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]
        return CandidateMemory(
            id=f"cand_ext_{digest}",
            type=extracted.type,
            claim=redact_text(extracted.claim),
            proposed_destination=extracted.proposed_destination,
            confidence=extracted.confidence,
            importance=extracted.importance,
            promotion_reason=extracted.promotion_reason,
            source_refs=list(extracted.source_refs),
            gate_decision=GateDecision.PENDING,
        )

    def _upsert_candidate(self, store: MemoryV2Store, index: MemoryV2Index, candidate: CandidateMemory) -> str:
        if self._matches_existing_active_memory(store, candidate):
            return "duplicate_existing_memory"
        candidates = store.list_candidates()
        key = self._candidate_key(candidate)
        for existing in candidates:
            if self._candidate_key(existing) != key:
                continue
            if existing.gate_decision != GateDecision.PENDING:
                return "duplicate_decided_candidate"
            merged_refs = list(existing.source_refs)
            for source_ref in candidate.source_refs:
                if source_ref not in merged_refs:
                    merged_refs.append(source_ref)
            if merged_refs == list(existing.source_refs):
                return "duplicate_candidate"
            data = existing.to_dict()
            data["source_refs"] = merged_refs
            merged = CandidateMemory.from_dict(data)
            store.rewrite_candidates([merged if item.id == existing.id else item for item in candidates])
            index.index_candidate(merged)
            candidate.id = merged.id
            return "merged"
        store.append_candidate(candidate)
        index.index_candidate(candidate)
        return "created"

    def _matches_existing_active_memory(self, store: MemoryV2Store, candidate: CandidateMemory) -> bool:
        candidate_norm = self._normalize_claim(candidate.claim)
        candidate_type = getattr(candidate.type, "value", str(candidate.type))
        for item in store.list_memory_items(status=MemoryStatus.ACTIVE.value):
            item_type = getattr(item.type, "value", str(item.type))
            if item_type != candidate_type:
                continue
            values = [item.value, item.summary, item.body]
            if any(self._normalize_claim(str(value or "")) == candidate_norm for value in values):
                return True
        return False

    def _candidate_key(self, candidate: CandidateMemory) -> tuple[str, str, str]:
        candidate_type = getattr(candidate.type, "value", str(candidate.type))
        return self._dedupe_key_parts(candidate_type, candidate.proposed_destination, candidate.claim)

    def _dedupe_key_parts(self, candidate_type: str, destination: str, claim: str) -> tuple[str, str, str]:
        return (str(candidate_type), str(destination).strip().lower(), self._normalize_claim(claim))

    @staticmethod
    def _event_user_text(event: Dict[str, Any]) -> str:
        return str(event.get("user_content") or "").strip()

    @classmethod
    def _clean_text(cls, text: str) -> str:
        cleaned = cls._SENDER_PREFIX_RE.sub("", str(text or "")).strip()
        return redact_text(cleaned)

    @classmethod
    def _should_skip_text(cls, text: str) -> bool:
        if not text or len(text) < 8:
            return True
        if contains_sensitive_text(text) or "[REDACTED" in text:
            return True
        lowered = text.lower()
        if "api key" in lowered or "password" in lowered or "token" in lowered or "secret" in lowered:
            return True
        if text.endswith("?"):
            return True
        if cls._INSTRUCTION_BAIT_RE.search(text):
            return True
        # Project next-actions and explicit follow-ups are allowed even when they
        # mention tomorrow; generic time-bound chatter is not.
        if cls._SCOPED_CURRENT_RE.search(text):
            return True
        if cls._EPHEMERAL_RE.search(text) and not (cls._PROJECT_FIELD_RE.match(text) or any(p.match(text) for p in cls._OPEN_LOOP_PATTERNS)):
            return True
        return False

    @staticmethod
    def _finish_sentence(text: str) -> str:
        value = re.sub(r"\s+", " ", str(text or "").strip())
        value = value.rstrip()
        if value and value[-1] not in ".!?":
            value += "."
        return value

    @staticmethod
    def _normalize_claim(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\[\] ]+", " ", str(text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _source_exists(store: MemoryV2Store, source_id: str) -> bool:
        return store.source_ref_exists(source_id)
