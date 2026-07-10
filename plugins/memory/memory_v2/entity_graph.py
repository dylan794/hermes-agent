"""Lightweight deterministic entity/graph drafts for Memory v2.

This module intentionally builds derived, report-only graph views from already
canonical Memory v2 records. It does not store graph state and it avoids copying
raw claim/body text into graph edges.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .report_safety import report_safe_serialize

_KNOWN_ENTITIES = (
    ("Memory v2", "memory_system"),
    ("LoCoMo", "memory_eval"),
    ("Hermes", "assistant_platform"),
    ("Qwen", "model_family"),
    ("QQQ", "market_index"),
    ("Nasdaq", "market_index"),
    ("TTS", "voice_tech"),
    ("the user", "user_person"),
)
_STOP_TITLE = {"Project", "Memory", "Fact", "Preference", "Belief", "Environment", "The", "This", "That"}


@dataclass(frozen=True)
class EntityLink:
    entity_id: str
    label: str
    bucket: str
    record_id: str
    record_type: str
    relation: str
    source_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "bucket": self.bucket,
            "record_id": self.record_id,
            "record_type": self.record_type,
            "relation": self.relation,
            "source_refs": list(self.source_refs),
        }


def extract_entity_links(record: Any, *, max_entities: int = 8) -> list[EntityLink]:
    """Extract compact entity links from a MemoryItem/ProjectCard/CandidateMemory.

    The extractor is deliberately cheap and deterministic: known product/project
    names plus title-case/acronym tokens are detected, but returned identifiers
    and display labels are safe derived metadata. Raw entity text is never copied
    into ids, labels, or record endpoints.
    """

    raw_record_id = str(getattr(record, "id", "") or "")
    record_id = _stable_key("record", f"{_record_type(record)}:{raw_record_id}")
    record_type = _record_type(record)
    text = _record_text(record)
    source_refs = tuple(str(ref) for ref in getattr(record, "source_refs", []) or [])
    relation = _relation_for_record(record)

    entities: list[tuple[str, str, str]] = []
    lowered = text.lower()
    for label, bucket in _KNOWN_ENTITIES:
        if label.lower() in lowered:
            entities.append((label, _stable_key("entity", f"known:{label.lower()}"), bucket))

    if not hasattr(record, "claim"):
        for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9_+.-]{2,}|[A-Z]{2,})\b", text):
            label = match.group(0).strip(".,:;()[]{}")
            if label in _STOP_TITLE or not label or len(label) > 40:
                continue
            entity_id = _stable_key("entity", f"derived:{_slug(label)}")
            if (label, entity_id, "derived_token") not in entities:
                entities.append((label, entity_id, "derived_token"))

    links: list[EntityLink] = []
    seen: set[str] = set()
    for _raw_label, entity_id, bucket in entities:
        if entity_id in seen:
            continue
        seen.add(entity_id)
        links.append(
            EntityLink(
                entity_id=entity_id,
                label=bucket,
                bucket=bucket,
                record_id=record_id,
                record_type=record_type,
                relation=relation,
                source_refs=source_refs,
            )
        )
        if len(links) >= max(1, int(max_entities)):
            break
    return links


def build_entity_graph_draft(
    *,
    memory_items: Iterable[Any] = (),
    project_cards: Iterable[Any] = (),
    candidates: Iterable[Any] = (),
    max_entities_per_record: int = 8,
    max_records: int = 500,
    max_entities: int = 200,
    max_edges: int = 500,
) -> dict[str, Any]:
    """Build a JSON-serializable draft graph from Memory v2 records.

    This is a derived report, not a persisted graph index. Edges intentionally
    omit raw memory text and contain only source ids for grounding.
    """

    records = [*memory_items, *project_cards, *candidates][: max(0, int(max_records))]
    max_entities = max(1, int(max_entities))
    max_edges = max(1, int(max_edges))
    entity_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    records_with_entities = 0

    for record in records:
        links = extract_entity_links(record, max_entities=max_entities_per_record)
        if links:
            records_with_entities += 1
        for link in links:
            if link.entity_id not in entity_by_id and len(entity_by_id) >= max_entities:
                continue
            entity_by_id.setdefault(link.entity_id, {"id": link.entity_id, "label": link.label, "bucket": link.bucket})
            edge_key = (link.entity_id, link.record_id, link.relation)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            if len(edges) >= max_edges:
                continue
            edges.append(
                {
                    "from": link.entity_id,
                    "to": link.record_id,
                    "relation": link.relation,
                    "record_type": link.record_type,
                    "source_refs": list(link.source_refs),
                }
            )

    entities = sorted(entity_by_id.values(), key=lambda item: item["id"])
    edges.sort(key=lambda item: (item["from"], item["to"], item["relation"]))
    return report_safe_serialize({
        "version": 1,
        "status": "draft",
        "policy": "report_only_no_mutation",
        "entities": entities,
        "edges": edges,
        "summary": {
            "entity_count": len(entities),
            "edge_count": len(edges),
            "records_considered": len(records),
            "records_with_entities": records_with_entities,
        },
    })


def _record_text(record: Any) -> str:
    parts = [
        getattr(record, "name", ""),
        getattr(record, "subject", ""),
        getattr(record, "predicate", ""),
        getattr(record, "value", ""),
        getattr(record, "summary", ""),
        getattr(record, "body", ""),
        getattr(record, "claim", ""),
        getattr(record, "goal", ""),
        getattr(record, "current_state", ""),
        " ".join(str(value) for value in (getattr(record, "decisions", []) or [])),
        " ".join(str(value) for value in (getattr(record, "open_questions", []) or [])),
        " ".join(str(value) for value in (getattr(record, "next_actions", []) or [])),
        " ".join(str(value) for value in (getattr(record, "related_entities", []) or [])),
    ]
    return "\n".join(str(part) for part in parts if part)


def _record_type(record: Any) -> str:
    if hasattr(record, "claim"):
        return "candidate"
    if hasattr(record, "name") and hasattr(record, "current_state"):
        return "project_card"
    value = getattr(record, "type", "memory_item")
    return str(getattr(value, "value", value))


def _relation_for_record(record: Any) -> str:
    if hasattr(record, "claim"):
        return "mentioned_in_candidate"
    if hasattr(record, "name") and hasattr(record, "current_state"):
        return "related_to_project"
    return "mentioned_in_memory"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "unknown"


def _stable_key(prefix: str, value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
