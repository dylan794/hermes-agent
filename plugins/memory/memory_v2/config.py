"""Centralized feature flags for Memory v2.

The defaults here are intentionally conservative: read-only/synthetic archive
inspection can be available, but mutating or autonomous behavior stays disabled
until the active Hermes profile opts in via ``config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ArchiveFlags:
    enabled: bool = True
    capture_enabled: bool = False
    backfill_enabled: bool = False
    # Archive evidence is private by default. Operators must explicitly opt
    # into each read surface after enabling the provider for a profile.
    search_tools_enabled: bool = False
    show_tools_enabled: bool = False
    # Internal raw-evidence hydration for exact/deep prefetch remains a separate
    # fail-closed opt-in from model-visible archive tools.
    prefetch_raw_enabled: bool = False
    # When capture is enabled, preserve bounded/redacted tool results as raw
    # evidence and pending work-episode candidates.
    include_tool_outputs: bool = False


@dataclass(frozen=True)
class ExtractionFlags:
    enabled: bool = False
    candidate_creation_enabled: bool = False
    small_model_enabled: bool = False


@dataclass(frozen=True)
class ConsolidationFlags:
    enabled: bool = False


@dataclass(frozen=True)
class PrefetchFlags:
    enabled: bool = False


@dataclass(frozen=True)
class ReviewApplyFlags:
    enabled: bool = False


@dataclass(frozen=True)
class AutoPromoteFlags:
    enabled: bool = False


@dataclass(frozen=True)
class ContradictionFlags:
    auto_supersede: bool = False
    create_candidates: bool = False


@dataclass(frozen=True)
class WorkingMemoryFlags:
    enabled: bool = False


@dataclass(frozen=True)
class MemoryV2FeatureFlags:
    archive: ArchiveFlags = field(default_factory=ArchiveFlags)
    extraction: ExtractionFlags = field(default_factory=ExtractionFlags)
    consolidation: ConsolidationFlags = field(default_factory=ConsolidationFlags)
    prefetch: PrefetchFlags = field(default_factory=PrefetchFlags)
    review_apply: ReviewApplyFlags = field(default_factory=ReviewApplyFlags)
    auto_promote: AutoPromoteFlags = field(default_factory=AutoPromoteFlags)
    contradictions: ContradictionFlags = field(default_factory=ContradictionFlags)
    working_memory: WorkingMemoryFlags = field(default_factory=WorkingMemoryFlags)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool(section: Mapping[str, Any], name: str, default: bool) -> bool:
    value = section.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def load_memory_v2_config(hermes_home: str | Path | None) -> MemoryV2FeatureFlags:
    """Load ``memory_v2`` feature flags from the active Hermes profile config.

    Missing files, malformed YAML, and malformed sections all fall back to safe
    defaults. This function never consults a hardcoded ``~/.hermes`` path; the
    caller must provide the profile's ``hermes_home`` when it is known.
    """

    defaults = MemoryV2FeatureFlags()
    if not hermes_home:
        return defaults
    config_path = Path(hermes_home).expanduser().resolve() / "config.yaml"
    if not config_path.exists():
        return defaults
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    root = _mapping(loaded)
    section = _mapping(root.get("memory_v2"))
    archive = _mapping(section.get("archive"))
    extraction = _mapping(section.get("extraction"))
    consolidation = _mapping(section.get("consolidation"))
    prefetch = _mapping(section.get("prefetch"))
    review_apply = _mapping(section.get("review_apply"))
    auto_promote = _mapping(section.get("auto_promote"))
    contradictions = _mapping(section.get("contradictions"))
    working_memory = _mapping(section.get("working_memory"))

    extraction_enabled = _bool(extraction, "enabled", defaults.extraction.enabled)
    candidate_default = defaults.extraction.candidate_creation_enabled

    return MemoryV2FeatureFlags(
        archive=ArchiveFlags(
            enabled=_bool(archive, "enabled", defaults.archive.enabled),
            capture_enabled=_bool(archive, "capture_enabled", defaults.archive.capture_enabled),
            backfill_enabled=_bool(archive, "backfill_enabled", defaults.archive.backfill_enabled),
            search_tools_enabled=_bool(archive, "search_tools_enabled", defaults.archive.search_tools_enabled),
            show_tools_enabled=_bool(archive, "show_tools_enabled", defaults.archive.show_tools_enabled),
            prefetch_raw_enabled=_bool(archive, "prefetch_raw_enabled", defaults.archive.prefetch_raw_enabled),
            include_tool_outputs=_bool(archive, "include_tool_outputs", defaults.archive.include_tool_outputs),
        ),
        extraction=ExtractionFlags(
            enabled=extraction_enabled,
            candidate_creation_enabled=_bool(
                extraction,
                "candidate_creation_enabled",
                candidate_default,
            ),
            small_model_enabled=_bool(
                extraction,
                "small_model_enabled",
                defaults.extraction.small_model_enabled,
            ),
        ),
        consolidation=ConsolidationFlags(
            enabled=_bool(consolidation, "enabled", defaults.consolidation.enabled),
        ),
        prefetch=PrefetchFlags(
            enabled=_bool(prefetch, "enabled", defaults.prefetch.enabled),
        ),
        review_apply=ReviewApplyFlags(
            enabled=_bool(review_apply, "enabled", defaults.review_apply.enabled),
        ),
        auto_promote=AutoPromoteFlags(
            enabled=_bool(auto_promote, "enabled", defaults.auto_promote.enabled),
        ),
        contradictions=ContradictionFlags(
            auto_supersede=_bool(
                contradictions,
                "auto_supersede",
                defaults.contradictions.auto_supersede,
            ),
            create_candidates=_bool(
                contradictions,
                "create_candidates",
                defaults.contradictions.create_candidates,
            ),
        ),
        working_memory=WorkingMemoryFlags(
            enabled=_bool(working_memory, "enabled", defaults.working_memory.enabled),
        ),
    )
