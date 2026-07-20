"""Offline, source-grounded, candidate-only extraction for Memory v2."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .index import MemoryV2Index
from .redaction import contains_sensitive_text, escape_untrusted_evidence_text, redact_text
from .schemas import (
    CandidateClaimKind,
    CandidateMemory,
    GateDecision,
    MemoryStatus,
    MemoryType,
    ValidationError,
    normalize_project_id,
)
from .store import MemoryV2Store

EvidenceSpan = Dict[str, Any]
StructuredModelAdapter = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class ExtractionReport:
    considered_events: int = 0
    created: int = 0
    merged: int = 0
    skipped: int = 0
    model_created: int = 0
    model_merged: int = 0
    model_rejected: int = 0
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
            "model_created": self.model_created,
            "model_merged": self.model_merged,
            "model_rejected": self.model_rejected,
            "created_ids": list(self.created_ids),
            "merged_ids": list(self.merged_ids),
            "skipped_reasons": dict(self.skipped_reasons),
        }


@dataclass(frozen=True)
class _ExtractedCandidate:
    type: str
    claim_kind: str
    claim: str
    proposed_destination: str
    promotion_reason: str
    confidence: float
    durability: float
    importance: float
    source_refs: List[str]
    evidence_spans: List[EvidenceSpan]
    negative_evidence: List[EvidenceSpan] = field(default_factory=list)
    extraction_method: str = "deterministic"


class OfflineSessionExtractor:
    """Extract pending candidates from bounded canonical session evidence.

    The deterministic pass is always available. A caller may optionally provide
    a small-model adapter, but model rows pass the same exact-span, source,
    authority, and candidate-only validation before any candidate is written.
    """

    version = "offline_extraction:v2"
    model_schema_version = 1

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
    _DECISION_PATTERNS = (
        re.compile(r"\b(?:we|i)\s+decided\s+to\s+(?P<value>.+?)(?:\.|$)", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bthe\s+decision\s+is\s+to\s+(?P<value>.+?)(?:\.|$)", re.IGNORECASE | re.DOTALL),
        re.compile(r"\b(?:let's|we\s+will)\s+(?P<value>use|switch|keep|remove|add|adopt)\s+(?P<rest>.+?)(?:\.|$)", re.IGNORECASE | re.DOTALL),
    )
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
    _ARTIFACT_AUTHORITY_RE = re.compile(
        r"(?P<value>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]+\s+is\s+(?:now\s+)?(?:the\s+)?(?:authoritative\s+source|source\s+of\s+truth).+?)(?:$|\n)",
        re.IGNORECASE,
    )
    _USER_BLOCKER_RE = re.compile(
        r"\b(?:the\s+blocker\s+is|we(?:'re|\s+are)\s+blocked\s+by|i\s+can't\s+continue\s+because)\s+(?P<value>.+?)(?:\.|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _USER_COMPLETED_RE = re.compile(
        r"\b(?:we|i)\s+(?:finished|completed|fixed)\s+(?P<value>.+?)(?:\.|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _NEGATED_COMPLETION_RE = re.compile(
        r"\b(?:did\s+not|didn't|never|not)\s+(?:finish|finished|complete|completed|fix|fixed)\b"
        r"|\b(?:finished|completed|fixed)\s+(?:neither|nothing|none)\b"
        r"|\b(?:finished|completed|fixed)\b[^.\n]*(?:\bbut\b|\bhowever\b)[^.\n]*\b(?:fail(?:ed|ure)?|broken|incomplete|not\s+done)\b",
        re.IGNORECASE,
    )
    _USER_CONTRADICTION_RE = re.compile(
        r"\b(?:the\s+test|testing|the\s+result)\s+(?:showed|proved|confirmed)\s+(?P<value>.+?\s+(?:wrong|false|invalid))(?:\.|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _CONSTRAINT_RE = re.compile(
        r"\b(?:must|should|needs?\s+to|is\s+required\s+to|remains?|stays?|cannot|can't)\b",
        re.IGNORECASE,
    )
    _PROPOSAL_RE = re.compile(
        r"(?P<value>(?:I\s+propose|I\s+recommend|We\s+should|Let's)\s+.+?)(?:\n|$)",
        re.IGNORECASE,
    )
    _ACCEPTANCE_RE = re.compile(
        r"^\s*(?:do\s+it|go\s+ahead|sounds\s+good|that\s+works|yes[,!. ]|okay[,!. ]|ok[,!. ]|approved\b)",
        re.IGNORECASE,
    )
    _REJECTION_RE = re.compile(r"\b(?:don't|do\s+not|no[,!. ]|reject|not\s+approved)\b", re.IGNORECASE)
    _TEST_FAILURE_RE = re.compile(r"\b(failed|assertionerror|test\s+failed|assumption\s+was\s+wrong|proved\s+.+?\s+(?:wrong|false))\b", re.IGNORECASE)
    _NEGATED_FAILURE_RE = re.compile(r"\b(?:0\s+failed|no\s+(?:tests?\s+)?failures?|without\s+failures?)\b", re.IGNORECASE)
    _BLOCKER_RE = re.compile(r"\b(error|exception|modulenotfounderror|blocked|cannot|permission denied|timed?\s*out)\b", re.IGNORECASE)
    _NEGATED_BLOCKER_RE = re.compile(r"\b(?:0\s+errors?|no\s+errors?|without\s+errors?|not\s+blocked)\b", re.IGNORECASE)
    _COMPLETED_RE = re.compile(r"\b(?:\d+\s+passed|all\s+tests\s+passed|completed|created|written|success(?:ful(?:ly)?)?)\b", re.IGNORECASE)
    _ENV_TOOL_RE = re.compile(r"\b(?:microsoft-standard-wsl|wsl2?|gnu/linux|python\s+\d+\.\d+|ubuntu\s+\d+|windows\s+nt)\b", re.IGNORECASE)
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

    def __init__(self, *, model_adapter: Optional[StructuredModelAdapter] = None) -> None:
        self.model_adapter = model_adapter

    @classmethod
    def user_evidence_supports_completion(cls, text: str) -> bool:
        """Return whether user evidence asserts, rather than negates, completion."""
        return bool(
            cls._USER_COMPLETED_RE.search(str(text or ""))
            and not cls._NEGATED_COMPLETION_RE.search(str(text or ""))
        )

    @classmethod
    def classify_tool_outcome(cls, text: str) -> str:
        """Classify bounded tool evidence with failures/blockers taking precedence."""
        value = str(text or "")
        has_failure = bool(
            cls._TEST_FAILURE_RE.search(value)
            and not cls._NEGATED_FAILURE_RE.search(value)
        )
        has_blocker = bool(
            cls._BLOCKER_RE.search(value)
            and not cls._NEGATED_BLOCKER_RE.search(value)
        )
        if has_failure:
            return "failed"
        if has_blocker:
            return "blocked"
        if cls._COMPLETED_RE.search(value):
            return "completed"
        return "observed"

    def extract(
        self,
        store: MemoryV2Store,
        index: MemoryV2Index,
        *,
        session_id: str = "",
        recent_raw_limit: Optional[int] = None,
        use_model: bool = False,
    ) -> ExtractionReport:
        report = ExtractionReport()
        events = list(self._events(store, session_id=session_id, limit=recent_raw_limit))
        event_map = {str(event.get("id") or ""): event for event in events}
        deterministic: List[_ExtractedCandidate] = []
        for event in events:
            report.considered_events += 1
            event_id = str(event.get("id") or "").strip()
            if not event_id:
                report.skip("empty_event")
                continue
            rows = self._extract_event(event)
            if not rows:
                report.skip("no_durable_candidate")
            deterministic.extend(rows)
        deterministic.extend(self._accepted_proposals(events))
        self._persist_rows(store, index, deterministic, report)

        if use_model:
            if self.model_adapter is None:
                report.skip("model_adapter_unavailable")
            else:
                try:
                    response = self.model_adapter(self._model_payload(events))
                    model_rows, rejected = self._validate_model_response(response, event_map, store)
                    report.model_rejected += rejected
                    self._persist_rows(store, index, model_rows, report, model=True)
                except Exception:
                    report.model_rejected += 1
                    report.skip("model_adapter_failed_closed")
        return report

    def _events(self, store: MemoryV2Store, *, session_id: str, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
        safe_limit = 50 if limit is None else int(limit)
        try:
            events = store.search_raw_events("", session_id=str(session_id or ""), limit=safe_limit)
        except ValidationError:
            return []
        return sorted(events, key=lambda event: (int(event.get("chain_index") or 0), str(event.get("created_at") or "")))

    def _extract_event(self, event: Dict[str, Any]) -> List[_ExtractedCandidate]:
        event_type = str(event.get("type") or "").lower()
        if event_type == "turn":
            return self._extract_turn(event)
        if event_type == "tool":
            return self._extract_tool(event)
        return []

    def _extract_turn(self, event: Dict[str, Any]) -> List[_ExtractedCandidate]:
        source_id = str(event.get("id") or "")
        raw_text = str(event.get("user_content") or "")
        clean = self._clean_text(raw_text)
        if not clean or self._EXPLICIT_MEMORY_RE.match(clean) or self._should_skip_text(clean):
            return []
        span = self._span(event, "user_content", raw_text, role="user")
        if span is None:
            return []
        rows: List[_ExtractedCandidate] = []
        if match := self._PROJECT_FIELD_RE.match(clean):
            rows.append(self._project_candidate(match, source_id=source_id, span=span))
            return rows
        if match := self._ENV_RE.match(clean):
            value = self._finish_sentence(match.group("value"))
            rows.append(self._row(MemoryType.ENVIRONMENT, "environment_state", f"Hermes environment fact: {value}", "semantic/items", 0.78, 0.88, 0.64, [span]))
        for pattern in self._PREFERENCE_PATTERNS:
            if match := pattern.match(clean):
                value = self._finish_sentence(match.group("value"))
                rows.append(self._row(MemoryType.PREFERENCE, "preference", f"User prefers {value}", "semantic/items", 0.8, 0.86, 0.68, [span]))
                break
        for pattern in self._DECISION_PATTERNS:
            if match := pattern.search(clean):
                value = match.groupdict().get("value") or ""
                if match.groupdict().get("rest"):
                    value = f"{value} {match.group('rest')}"
                decision_span = self._span(event, "user_content", match.group(0), role="user") or span
                rows.append(self._row(MemoryType.PROJECT_STATE, "decision", f"Decision: {self._finish_sentence(value)}", "semantic/items", 0.8, 0.9, 0.76, [decision_span]))
                break
        if match := self._ARTIFACT_AUTHORITY_RE.search(clean):
            artifact_span = self._span(event, "user_content", match.group("value"), role="user") or span
            rows.append(self._row(MemoryType.PROJECT_STATE, "authoritative_artifact", f"Authoritative artifact: {self._finish_sentence(match.group('value'))}", "semantic/items", 0.84, 0.96, 0.82, [artifact_span]))
        if match := self._USER_BLOCKER_RE.search(clean):
            blocker_span = self._span(event, "user_content", match.group(0), role="user") or span
            rows.append(self._row(MemoryType.PROJECT_STATE, "blocker", f"Blocker: {self._finish_sentence(match.group('value'))}", "working/open_loops.yaml", 0.78, 0.72, 0.78, [blocker_span]))
        for match in self._USER_COMPLETED_RE.finditer(clean):
            completion_text = match.group(0)
            if not self.user_evidence_supports_completion(completion_text):
                continue
            completed_span = self._span(event, "user_content", completion_text, role="user") or span
            rows.append(self._row(MemoryType.EPISODE, "completed_action", f"Completed action: {self._finish_sentence(match.group('value'))}", "semantic/items", 0.78, 0.58, 0.68, [completed_span]))
        if match := self._USER_CONTRADICTION_RE.search(clean):
            negative_span = self._span(event, "user_content", match.group(0), role="user") or span
            rows.append(self._row(MemoryType.EPISODE, "contradiction", f"Contradiction candidate: {self._finish_sentence(match.group('value'))}", "semantic/items", 0.8, 0.78, 0.76, [negative_span], negative=[negative_span]))
        if self._SKILL_RE.search(clean):
            rows.append(self._row(MemoryType.PROCEDURE_REF, "skill_candidate", self._finish_sentence(clean), "skills", 0.7, 0.85, 0.72, [span]))
        for pattern in self._OPEN_LOOP_PATTERNS:
            if match := pattern.match(clean):
                rows.append(self._row(MemoryType.PROJECT_STATE, "open_loop", f"Open loop: {self._finish_sentence(match.group('value'))}", "working/open_loops.yaml", 0.68, 0.65, 0.64, [span]))
                break
        return rows

    def _extract_tool(self, event: Dict[str, Any]) -> List[_ExtractedCandidate]:
        field_name, raw_text = self._tool_evidence_field(event)
        clean = redact_text(raw_text).strip()
        if not clean or self._should_skip_tool_text(clean):
            return []
        span = self._span(event, field_name, raw_text, role="tool")
        if span is None:
            return []
        tool = redact_text(str(event.get("tool") or "tool"))[:80]
        bounded = self._finish_sentence(re.sub(r"\s+", " ", clean)[:700])
        rows: List[_ExtractedCandidate] = []
        outcome = self.classify_tool_outcome(clean)
        has_failure = outcome == "failed"
        has_blocker = outcome == "blocked"
        if has_failure:
            rows.append(self._row(MemoryType.EPISODE, "contradiction", f"Contradiction candidate from {tool}: {bounded}", "episodic/tool-results", 0.82, 0.78, 0.78, [span], negative=[span]))
        if has_blocker:
            rows.append(self._row(MemoryType.PROJECT_STATE, "blocker", f"Blocker observed in {tool}: {bounded}", "working/open_loops.yaml", 0.86, 0.72, 0.8, [span]))
        if self._ENV_TOOL_RE.search(clean):
            rows.append(self._row(MemoryType.ENVIRONMENT, "environment_state", f"Environment state established by {tool}: {bounded}", "semantic/items", 0.9, 0.9, 0.72, [span]))
        if outcome == "completed":
            rows.append(self._row(MemoryType.EPISODE, "completed_action", f"Completed action verified by {tool}: {bounded}", "episodic/tool-results", 0.9, 0.62, 0.7, [span]))
        return rows

    def _accepted_proposals(self, events: List[Dict[str, Any]]) -> List[_ExtractedCandidate]:
        rows: List[_ExtractedCandidate] = []
        previous_turn: Optional[Dict[str, Any]] = None
        for event in events:
            if str(event.get("type") or "") != "turn":
                continue
            user_text = str(event.get("user_content") or "")
            user_clean = redact_text(user_text)
            accepted = bool(self._ACCEPTANCE_RE.search(user_clean) and not self._REJECTION_RE.search(user_clean))
            if previous_turn is not None and accepted and not self._unsafe_evidence_text(user_text):
                assistant_text = str(previous_turn.get("assistant_content") or "")
                if self._unsafe_evidence_text(assistant_text):
                    previous_turn = event
                    continue
                if proposal := self._PROPOSAL_RE.search(assistant_text):
                    proposal_span = self._span(previous_turn, "assistant_content", proposal.group("value"), role="assistant")
                    acceptance_span = self._span(event, "user_content", user_text, role="user")
                    if proposal_span and acceptance_span:
                        claim = re.sub(r"^(?:I\s+propose|I\s+recommend|We\s+should|Let's)\s+", "", proposal.group("value"), flags=re.IGNORECASE)
                        rows.append(self._row(MemoryType.PROJECT_STATE, "accepted_proposal", f"Accepted proposal: {self._finish_sentence(claim)}", "semantic/items", 0.78, 0.84, 0.74, [proposal_span, acceptance_span]))
            previous_turn = event
        return rows

    def _row(
        self,
        memory_type: MemoryType,
        claim_kind: str,
        claim: str,
        destination: str,
        confidence: float,
        durability: float,
        importance: float,
        spans: List[EvidenceSpan],
        *,
        negative: Optional[List[EvidenceSpan]] = None,
        extraction_method: str = "deterministic",
    ) -> _ExtractedCandidate:
        refs = list(dict.fromkeys(str(span["source_id"]) for span in spans))
        return _ExtractedCandidate(
            type=memory_type.value,
            claim_kind=claim_kind,
            claim=claim,
            proposed_destination=destination,
            promotion_reason=f"{self.version} {claim_kind}: source-grounded candidate; pending review required.",
            confidence=confidence,
            durability=durability,
            importance=importance,
            source_refs=refs,
            evidence_spans=spans,
            negative_evidence=list(negative or []),
            extraction_method=extraction_method,
        )

    def _project_candidate(self, match: re.Match[str], *, source_id: str, span: EvidenceSpan) -> _ExtractedCandidate:
        project = self._finish_sentence(match.group("project")).rstrip(".")
        if project.lower().startswith("project "):
            project = project[len("project ") :].strip()
        kind = str(match.group("kind") or "").lower().replace(" ", "_")
        value = self._finish_sentence(match.group("value"))
        slug = normalize_project_id(project).split(":", 1)[1]
        return self._row(MemoryType.PROJECT_STATE, kind, f"Project {project} {kind.replace('_', ' ')}: {value}", f"semantic/projects/{slug}.yaml", 0.78, 0.86, 0.72, [span])

    def _persist_rows(
        self,
        store: MemoryV2Store,
        index: MemoryV2Index,
        rows: List[_ExtractedCandidate],
        report: ExtractionReport,
        *,
        model: bool = False,
    ) -> None:
        for extracted in rows:
            if not all(self._source_exists(store, ref, index=index) for ref in extracted.source_refs):
                report.skip("missing_source_ref")
                if model:
                    report.model_rejected += 1
                continue
            candidate = self._candidate_from_extracted(extracted)
            result = self._upsert_candidate(store, index, candidate)
            if result == "created":
                report.created += 1
                report.created_ids.append(candidate.id)
                if model:
                    report.model_created += 1
            elif result == "merged":
                report.merged += 1
                report.merged_ids.append(candidate.id)
                if model:
                    report.model_merged += 1
            else:
                report.skip(result)

    def _candidate_from_extracted(self, extracted: _ExtractedCandidate) -> CandidateMemory:
        normalized = self._dedupe_key_parts(extracted.type, extracted.proposed_destination, extracted.claim_kind, extracted.claim)
        digest = hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]
        return CandidateMemory(
            id=f"cand_ext_{digest}",
            type=extracted.type,
            claim=redact_text(extracted.claim),
            claim_kind=extracted.claim_kind,
            proposed_destination=extracted.proposed_destination,
            confidence=extracted.confidence,
            durability=extracted.durability,
            importance=extracted.importance,
            promotion_reason=extracted.promotion_reason,
            source_refs=list(extracted.source_refs),
            evidence_spans=[dict(span) for span in extracted.evidence_spans],
            negative_evidence=[dict(span) for span in extracted.negative_evidence],
            extraction_method=extracted.extraction_method,
            extractor_version=self.version,
            gate_decision=GateDecision.PENDING,
        )

    def _upsert_candidate(self, store: MemoryV2Store, index: MemoryV2Index, candidate: CandidateMemory) -> str:
        with store.profile_lock():
            return self._upsert_candidate_locked(store, index, candidate)

    def _upsert_candidate_locked(self, store: MemoryV2Store, index: MemoryV2Index, candidate: CandidateMemory) -> str:
        if self._matches_existing_active_memory(store, candidate):
            return "duplicate_existing_memory"
        candidates = store.list_candidates()
        for existing in candidates:
            if not self._equivalent_candidates(existing, candidate):
                continue
            if existing.gate_decision != GateDecision.PENDING:
                return "duplicate_decided_candidate"
            data = existing.to_dict()
            merged_refs = list(dict.fromkeys(list(existing.source_refs) + list(candidate.source_refs)))
            merged_spans = self._merge_spans(existing.evidence_spans, candidate.evidence_spans)
            merged_negative = self._merge_spans(existing.negative_evidence, candidate.negative_evidence)
            if merged_refs == list(existing.source_refs) and merged_spans == list(existing.evidence_spans) and merged_negative == list(existing.negative_evidence):
                return "duplicate_candidate"
            data.update({
                "source_refs": merged_refs,
                "evidence_spans": merged_spans,
                "negative_evidence": merged_negative,
                "confidence": max(existing.confidence, candidate.confidence),
                "durability": max(existing.durability, candidate.durability),
                "claim_kind": (
                    candidate.to_dict()["claim_kind"]
                    if existing.to_dict()["claim_kind"] == CandidateClaimKind.OTHER.value
                    else existing.to_dict()["claim_kind"]
                ),
                "extraction_method": (
                    candidate.extraction_method
                    if existing.extraction_method == "legacy"
                    else existing.extraction_method
                ),
                "extractor_version": existing.extractor_version or candidate.extractor_version,
            })
            merged = CandidateMemory.from_dict(data)
            store.rewrite_candidates([merged if item.id == existing.id else item for item in candidates])
            index.index_candidate(merged)
            candidate.id = merged.id
            return "merged"
        store.append_candidate(candidate)
        index.index_candidate(candidate)
        return "created"

    def _matches_existing_active_memory(self, store: MemoryV2Store, candidate: CandidateMemory) -> bool:
        candidate_type = getattr(candidate.type, "value", str(candidate.type))
        for item in store.list_memory_items(status=MemoryStatus.ACTIVE.value):
            if getattr(item.type, "value", str(item.type)) != candidate_type:
                continue
            if any(self._claims_similar(candidate.claim, str(value or "")) for value in (item.value, item.summary, item.body)):
                return True
        return False

    def _equivalent_candidates(self, left: CandidateMemory, right: CandidateMemory) -> bool:
        left_kind = left.to_dict()["claim_kind"]
        right_kind = right.to_dict()["claim_kind"]
        if left_kind != right_kind and CandidateClaimKind.OTHER.value not in {left_kind, right_kind}:
            return False
        left_type = getattr(left.type, "value", str(left.type))
        right_type = getattr(right.type, "value", str(right.type))
        if left_type != right_type:
            return False
        if str(left.proposed_destination).strip().lower() != str(right.proposed_destination).strip().lower():
            return False
        return self._claims_similar(left.claim, right.claim)

    @classmethod
    def _claims_similar(cls, left: str, right: str) -> bool:
        left_tokens = set(cls._semantic_tokens(left))
        right_tokens = set(cls._semantic_tokens(right))
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return overlap / union >= 0.82 or overlap / min(len(left_tokens), len(right_tokens)) >= 0.94

    @classmethod
    def _semantic_tokens(cls, text: str) -> List[str]:
        normalized = cls._normalize_claim(text)
        stop = {"decision", "project", "user", "prefers", "preference", "accepted", "proposal", "the", "a", "an", "to", "is", "was"}
        return [token for token in normalized.split() if token not in stop]

    def _dedupe_key_parts(self, candidate_type: str, destination: str, claim_kind: str, claim: str) -> tuple[str, str, str, str]:
        return (str(candidate_type), str(destination).strip().lower(), claim_kind, " ".join(self._semantic_tokens(claim)))

    @staticmethod
    def _merge_spans(left: List[EvidenceSpan], right: List[EvidenceSpan]) -> List[EvidenceSpan]:
        merged: List[EvidenceSpan] = []
        seen: set[tuple[Any, ...]] = set()
        for span in list(left) + list(right):
            key = (span.get("source_id"), span.get("field"), span.get("start"), span.get("end"), span.get("text"))
            if key not in seen:
                seen.add(key)
                merged.append(dict(span))
        return merged

    def _model_payload(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload_events: List[Dict[str, Any]] = []
        for event in events:
            fields: Dict[str, str] = {}
            for field_name in ("user_content", "assistant_content", "result", "content", "stdout", "stderr"):
                value = event.get(field_name)
                if isinstance(value, str) and value.strip():
                    escaped, _ = escape_untrusted_evidence_text(redact_text(value))
                    fields[field_name] = escaped[:4000]
            payload_events.append({
                "id": str(event.get("id") or ""),
                "type": str(event.get("type") or ""),
                "tool": redact_text(str(event.get("tool") or ""))[:120],
                "fields": fields,
            })
        return {
            "schema_version": self.model_schema_version,
            "instruction": "Return candidate claims only. Copy exact evidence spans. Assistant text is not user truth unless a user acceptance span is also cited. Never promote memory.",
            "events": payload_events,
        }

    def _validate_model_response(
        self,
        response: Dict[str, Any],
        events: Dict[str, Dict[str, Any]],
        store: MemoryV2Store,
    ) -> tuple[List[_ExtractedCandidate], int]:
        if not isinstance(response, dict) or response.get("version") != self.model_schema_version:
            return [], 1
        raw_candidates = response.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) > 25:
            return [], 1
        rows: List[_ExtractedCandidate] = []
        rejected = 0
        for raw in raw_candidates:
            try:
                if not isinstance(raw, dict):
                    raise ValidationError("model candidate must be an object")
                spans = self._validate_exact_spans(raw.get("evidence_spans"), events)
                negative = self._validate_exact_spans(raw.get("negative_evidence") or [], events)
                if not spans or not any(span["role"] in {"user", "tool"} for span in spans):
                    raise ValidationError("assistant-only model claims are not authoritative")
                if any(span["role"] == "assistant" for span in spans) and not any(
                    span["role"] == "user" and self._ACCEPTANCE_RE.search(span["text"]) for span in spans
                ):
                    raise ValidationError("assistant claims require exact user acceptance evidence")
                memory_type = MemoryType.coerce(raw.get("memory_type"), "memory_type")
                claim_kind_enum = CandidateClaimKind.coerce(raw.get("claim_kind"), "claim_kind")
                claim_kind = claim_kind_enum.value
                claim = redact_text(str(raw.get("claim") or "").strip())
                if not claim_kind or not claim or contains_sensitive_text(claim):
                    raise ValidationError("model claim is invalid")
                if not self._model_evidence_supports(claim_kind, spans, negative, events=events):
                    raise ValidationError("model evidence does not support the typed claim")
                if not self._model_claim_entailed(claim, claim_kind, spans):
                    raise ValidationError("model claim is not entailed by its exact evidence")
                confidence = float(raw.get("confidence"))
                durability = float(raw.get("durability"))
                if not 0.0 <= confidence <= 1.0 or not 0.0 <= durability <= 1.0:
                    raise ValidationError("model scores must be unit intervals")
                refs = list(dict.fromkeys(span["source_id"] for span in spans + negative))
                if not all(self._source_exists(store, ref) for ref in refs):
                    raise ValidationError("model source is not canonical")
                rows.append(self._row(
                    memory_type,
                    claim_kind,
                    claim,
                    self._destination_for(memory_type, claim_kind),
                    min(confidence, 0.82),
                    durability,
                    min(0.8, max(0.4, durability)),
                    spans,
                    negative=negative,
                    extraction_method="structured_model",
                ))
            except (KeyError, TypeError, ValueError, ValidationError):
                rejected += 1
        return rows, rejected

    def _model_evidence_supports(
        self,
        claim_kind: str,
        spans: List[EvidenceSpan],
        negative: List[EvidenceSpan],
        *,
        events: Dict[str, Dict[str, Any]],
    ) -> bool:
        user_texts = [span["text"] for span in spans if span["role"] == "user"]
        tool_texts = [span["text"] for span in spans if span["role"] == "tool"]
        if claim_kind == CandidateClaimKind.PREFERENCE.value:
            return any(any(pattern.match(text) for pattern in self._PREFERENCE_PATTERNS) for text in user_texts)
        if claim_kind == CandidateClaimKind.ENVIRONMENT_STATE.value:
            return any(self._ENV_RE.match(text) for text in user_texts) or any(self._ENV_TOOL_RE.search(text) for text in tool_texts)
        if claim_kind == CandidateClaimKind.DECISION.value:
            return any(any(pattern.search(text) for pattern in self._DECISION_PATTERNS) for text in user_texts)
        if claim_kind in {
            CandidateClaimKind.NEXT_ACTION.value,
            CandidateClaimKind.OPEN_QUESTION.value,
            CandidateClaimKind.CURRENT_STATE.value,
            CandidateClaimKind.GOAL.value,
            CandidateClaimKind.STATUS.value,
        }:
            return any(self._PROJECT_FIELD_RE.match(text) for text in user_texts)
        if claim_kind == CandidateClaimKind.CONSTRAINT.value:
            return any(self._CONSTRAINT_RE.search(text) for text in user_texts)
        if claim_kind == CandidateClaimKind.SKILL_CANDIDATE.value:
            return any(self._SKILL_RE.search(text) for text in user_texts)
        if claim_kind == CandidateClaimKind.OPEN_LOOP.value:
            return any(any(pattern.match(text) for pattern in self._OPEN_LOOP_PATTERNS) for text in user_texts)
        if claim_kind == CandidateClaimKind.BLOCKER.value:
            return any(self._USER_BLOCKER_RE.search(text) for text in user_texts) or any(
                self._BLOCKER_RE.search(text) and not self._NEGATED_BLOCKER_RE.search(text) for text in tool_texts
            )
        if claim_kind == CandidateClaimKind.COMPLETED_ACTION.value:
            return any(self.user_evidence_supports_completion(text) for text in user_texts) or any(
                self.classify_tool_outcome(text) == "completed" for text in tool_texts
            )
        if claim_kind == CandidateClaimKind.AUTHORITATIVE_ARTIFACT.value:
            return any(self._ARTIFACT_AUTHORITY_RE.search(text) for text in user_texts)
        if claim_kind == CandidateClaimKind.CONTRADICTION.value:
            negative_texts = [span["text"] for span in negative]
            return bool(negative_texts) and any(
                self._USER_CONTRADICTION_RE.search(text)
                or (self._TEST_FAILURE_RE.search(text) and not self._NEGATED_FAILURE_RE.search(text))
                for text in negative_texts
            )
        if claim_kind == CandidateClaimKind.ACCEPTED_PROPOSAL.value:
            proposal_spans = [
                span for span in spans
                if span["role"] == "assistant" and self._PROPOSAL_RE.search(span["text"])
            ]
            acceptance_spans = [
                span for span in spans
                if span["role"] == "user"
                and self._ACCEPTANCE_RE.search(span["text"])
                and not self._REJECTION_RE.search(span["text"])
            ]
            order = {event_id: position for position, event_id in enumerate(events)}
            for proposal in proposal_spans:
                proposal_event = events.get(proposal["source_id"], {})
                proposal_position = order.get(proposal["source_id"])
                if proposal_position is None:
                    continue
                for acceptance in acceptance_spans:
                    acceptance_event = events.get(acceptance["source_id"], {})
                    if order.get(acceptance["source_id"]) != proposal_position + 1:
                        continue
                    proposal_session = str(proposal_event.get("provider_session_id") or proposal_event.get("session_id") or "")
                    acceptance_session = str(acceptance_event.get("provider_session_id") or acceptance_event.get("session_id") or "")
                    if proposal_session and proposal_session == acceptance_session:
                        return True
            return False
        return False

    @classmethod
    def _model_claim_entailed(
        cls,
        claim: str,
        claim_kind: str,
        spans: List[EvidenceSpan],
    ) -> bool:
        """Conservatively require model claims to retain evidence content.

        Exact spans prove provenance, not meaning.  This lexical entailment gate
        intentionally rejects unsupported paraphrases rather than allowing a
        model to invert a preference, constraint, or decision.
        """
        aliases: Dict[str, str] = {
            "likes": "prefer", "like": "prefer", "prefers": "prefer", "preferred": "prefer",
            "concise": "short", "brief": "short", "verbose": "long",
            "answers": "answer", "responses": "answer", "response": "answer",
            "decided": "decide", "decision": "decide", "chooses": "choose", "chosen": "choose",
        }
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
            "i", "in", "is", "it", "me", "my", "of", "on", "or", "that", "the", "this", "to",
            "user", "we", "with", "will", "would", "claim", "accepted", "proposal",
        }

        def tokens(text: str) -> set[str]:
            values = re.findall(r"[a-z0-9]+", text.lower())
            return {aliases.get(value, value) for value in values if value not in stop and len(value) > 1}

        claim_tokens = tokens(claim)
        evidence_text = " ".join(span["text"] for span in spans)
        evidence_tokens = tokens(evidence_text)
        negation_re = re.compile(
            r"\b(?:not|no|never|without|cannot|can't|don't|doesn't|didn't|won't|wouldn't|shouldn't|isn't|aren't)\b"
            r"|\b(?:do|does|did|will|would|should|is|are)\s+not\b",
            re.IGNORECASE,
        )
        if bool(negation_re.search(claim)) != bool(negation_re.search(evidence_text)):
            return False
        if not claim_tokens:
            return False
        shared = claim_tokens & evidence_tokens
        required = 1 if len(claim_tokens) <= 2 else max(2, (len(claim_tokens) + 1) // 2)
        if len(shared) < required:
            return False
        # Opposite polarity/value markers must never be introduced by the model.
        opposites = ({"short", "long"}, {"enable", "disable"}, {"allow", "deny"}, {"keep", "delete"})
        for pair in opposites:
            if claim_tokens & pair and evidence_tokens & pair and (claim_tokens & pair) != (evidence_tokens & pair):
                return False
        return True

    def _validate_exact_spans(self, raw_spans: Any, events: Dict[str, Dict[str, Any]]) -> List[EvidenceSpan]:
        if not isinstance(raw_spans, list) or len(raw_spans) > 12:
            raise ValidationError("model evidence spans must be a bounded list")
        spans = CandidateMemory._validate_evidence_spans(raw_spans, "model_evidence_spans")
        validated: List[EvidenceSpan] = []
        for span in spans:
            event = events.get(str(span["source_id"]))
            if event is None:
                raise ValidationError("model evidence source is outside selected events")
            field_text = event.get(str(span["field"]))
            if not isinstance(field_text, str):
                raise ValidationError("model evidence field is absent")
            if contains_sensitive_text(field_text) or self._INSTRUCTION_BAIT_RE.search(field_text):
                raise ValidationError("model evidence field is unsafe")
            start, end = int(span["start"]), int(span["end"])
            if end > len(field_text) or field_text[start:end] != span["text"]:
                raise ValidationError("model evidence span is not an exact source slice")
            expected_role = "tool" if str(event.get("type") or "") == "tool" else (
                "assistant" if span["field"] == "assistant_content" else "user"
            )
            if span["role"] != expected_role:
                raise ValidationError("model evidence role does not match source field")
            validated.append(span)
        return validated

    @staticmethod
    def _destination_for(memory_type: MemoryType, claim_kind: str) -> str:
        if claim_kind == "blocker" or claim_kind == "open_loop":
            return "working/open_loops.yaml"
        if memory_type == MemoryType.PROCEDURE_REF:
            return "skills"
        return "semantic/items"

    @staticmethod
    def _tool_evidence_field(event: Dict[str, Any]) -> tuple[str, str]:
        for name in ("result", "stdout", "stderr", "content"):
            value = event.get(name)
            if isinstance(value, str) and value.strip():
                return name, value
        return "content", ""

    @staticmethod
    def _span(event: Dict[str, Any], field_name: str, selected_text: str, *, role: str) -> Optional[EvidenceSpan]:
        source_text = event.get(field_name)
        if not isinstance(source_text, str) or not selected_text:
            return None
        start = source_text.find(selected_text)
        if start < 0:
            return None
        return {
            "source_id": str(event.get("id") or ""),
            "field": field_name,
            "role": role,
            "start": start,
            "end": start + len(selected_text),
            "text": selected_text,
        }

    @classmethod
    def _clean_text(cls, text: str) -> str:
        return redact_text(cls._SENDER_PREFIX_RE.sub("", str(text or "")).strip())

    @classmethod
    def _should_skip_text(cls, text: str) -> bool:
        if not text or len(text) < 8 or contains_sensitive_text(text) or "[REDACTED" in text:
            return True
        lowered = text.lower()
        if any(marker in lowered for marker in ("api key", "password", "token", "secret")):
            return True
        if cls._INSTRUCTION_BAIT_RE.search(text) or cls._SCOPED_CURRENT_RE.search(text):
            return True
        if text.endswith("?"):
            return True
        if cls._EPHEMERAL_RE.search(text) and not (cls._PROJECT_FIELD_RE.match(text) or any(pattern.match(text) for pattern in cls._OPEN_LOOP_PATTERNS)):
            return True
        return False

    @classmethod
    def _unsafe_evidence_text(cls, text: str) -> bool:
        redacted = redact_text(str(text or ""))
        return bool(
            contains_sensitive_text(text)
            or "[REDACTED" in redacted
            or cls._INSTRUCTION_BAIT_RE.search(redacted)
        )

    @classmethod
    def _should_skip_tool_text(cls, text: str) -> bool:
        return bool(
            not text
            or contains_sensitive_text(text)
            or "[REDACTED" in text
            or cls._INSTRUCTION_BAIT_RE.search(text)
        )

    @staticmethod
    def _finish_sentence(text: str) -> str:
        value = re.sub(r"\s+", " ", str(text or "").strip()).rstrip()
        if value and value[-1] not in ".!?":
            value += "."
        return value

    @staticmethod
    def _normalize_claim(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\[\] ]+", " ", str(text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _source_exists(store: MemoryV2Store, source_id: str, *, index: Optional[MemoryV2Index] = None) -> bool:
        return store.source_ref_exists(source_id, index=index)
