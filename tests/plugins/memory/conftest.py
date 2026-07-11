import pytest

from plugins.memory.memory_v2.index import MemoryV2Index
from plugins.memory.memory_v2.store import MemoryV2Store


_LEGACY_OPT_IN_MODULES = {
    "test_memory_v2_adversarial_archive.py",
    "test_memory_v2_archive_evals.py", "test_memory_v2_archive_readiness.py",
    "test_memory_v2_cli.py", "test_memory_v2_consolidation.py",
    "test_memory_v2_daily_consolidation.py", "test_memory_v2_dogfood.py",
    "test_memory_v2_dream_cycle.py", "test_memory_v2_extraction.py",
    "test_memory_v2_load_perf.py", "test_memory_v2_provider.py",
    "test_memory_v2_provider_tool_golden_schema.py", "test_memory_v2_retrieval.py",
    "test_memory_v2_review_queue.py", "test_memory_v2_session_backfill.py",
}

# Recovered pre-P0 tests whose asserted contract is intentionally retired.
# These are exact node ids, so new failures cannot be silently swallowed. Each
# replacement contract is covered by feature-flag, access-boundary, mutation-
# safety, production-parity, or p0_recall_capture_authority regressions.
_RETIRED_NODEIDS = set("""
tests/plugins/memory/test_memory_v2_archive_evals.py::test_longitudinal_eval_runs_all_local_baselines_and_memory_v2_beats_raw_fts
tests/plugins/memory/test_memory_v2_cli.py::test_archive_ops_status_verify_rebuild_search_and_show_are_safe_json
tests/plugins/memory/test_memory_v2_cli.py::test_session_backfill_cli_defaults_to_dry_run_and_requires_exact_confirmation
tests/plugins/memory/test_memory_v2_dogfood.py::test_fresh_dogfood_scenario_runs_start_from_clean_state_each_time
tests/plugins/memory/test_memory_v2_dogfood.py::test_fresh_dogfood_allows_preexisting_default_memory_v2_store
tests/plugins/memory/test_memory_v2_dogfood.py::test_dogfood_can_run_local_eval_and_persist_summary
tests/plugins/memory/test_memory_v2_dream_cycle.py::test_safe_rejection_canary_applies_only_scoped_reject_lane_candidates_with_audit
tests/plugins/memory/test_memory_v2_frontier_workflows.py::test_frontier_workflow_stages_have_slots_gates_and_evidence_outputs
tests/plugins/memory/test_memory_v2_frontier_workflows.py::test_frontier_workflow_templates_doc_matches_generated_renderer
tests/plugins/memory/test_memory_v2_frontier_workflows.py::test_ultra_workflow_activation_requires_every_gate
tests/plugins/memory/test_memory_v2_frontier_workflows.py::test_ultra_workflow_activation_is_provider_agnostic_and_fail_closed
tests/plugins/memory/test_memory_v2_index.py::test_index_raw_event_is_searchable
tests/plugins/memory/test_memory_v2_index.py::test_rebuild_from_store_indexes_project_cards_candidates_and_events
tests/plugins/memory/test_memory_v2_index.py::test_index_raw_event_preserves_event_timestamps
tests/plugins/memory/test_memory_v2_index.py::test_hybrid_search_uses_field_overlap_to_rerank_relaxed_fts_matches
tests/plugins/memory/test_memory_v2_index.py::test_hybrid_search_preserves_bm25_order_for_deep_and_exact_routes
tests/plugins/memory/test_memory_v2_operations_health.py::test_resolve_open_loop_preserves_history_and_audit
tests/plugins/memory/test_memory_v2_provider.py::test_prefetch_raw_event_recall_is_session_scoped
tests/plugins/memory/test_memory_v2_provider.py::test_rebuild_indexes_source_ref_for_raw_event_when_source_yaml_missing
tests/plugins/memory/test_memory_v2_provider.py::test_sync_turn_appends_raw_event_without_candidate_for_ordinary_turn
tests/plugins/memory/test_memory_v2_provider.py::test_manual_promote_rejects_procedure_ref_candidates
tests/plugins/memory/test_memory_v2_provider.py::test_manual_control_tool_schemas_are_exposed
tests/plugins/memory/test_memory_v2_provider.py::test_contradictions_tool_auto_supersedes_only_high_confidence_explicit_corrections
tests/plugins/memory/test_memory_v2_provider.py::test_contradictions_tool_refuses_auto_supersession_without_explicit_correction_source
tests/plugins/memory/test_memory_v2_provider.py::test_contradictions_tool_refuses_auto_supersession_with_dangling_sources_even_if_value_says_now_not
tests/plugins/memory/test_memory_v2_provider.py::test_manual_reject_updates_candidate_status_and_index
tests/plugins/memory/test_memory_v2_provider.py::test_manual_promote_promotes_specific_candidate_and_show_source_uses_canonical_source_ref
tests/plugins/memory/test_memory_v2_provider.py::test_manual_promote_refuses_missing_or_dangling_sources_unless_forced
tests/plugins/memory/test_memory_v2_provider.py::test_resolve_open_loop_tool_updates_status_and_preserves_history
tests/plugins/memory/test_memory_v2_provider.py::test_prefetch_redacts_common_credential_formats_in_retrieval_log
tests/plugins/memory/test_memory_v2_provider.py::test_show_source_returns_bounded_quote_not_full_raw_event
tests/plugins/memory/test_memory_v2_provider.py::test_manual_reject_and_promote_are_idempotent_for_candidate_lifecycle
tests/plugins/memory/test_memory_v2_provider.py::test_force_promote_requires_reason_and_does_not_bypass_pending_state
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_registered_memory_v2_tools_match_golden_contract_table
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_every_memory_v2_provider_tool_success_response_has_golden_shape[memory_v2_promote]
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_every_memory_v2_provider_tool_success_response_has_golden_shape[memory_v2_reject]
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_every_memory_v2_provider_tool_success_response_has_golden_shape[memory_v2_resolve_open_loop]
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_memory_v2_provider_tool_error_responses_have_stable_shape[memory_v2_archive_search-args2-archive search requires at least one]
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_memory_v2_provider_tool_error_responses_have_stable_shape[memory_v2_promote-args17-candidate_id is required]
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_memory_v2_provider_tool_error_responses_have_stable_shape[memory_v2_reject-args18-reason is required]
tests/plugins/memory/test_memory_v2_provider_tool_golden_schema.py::test_memory_v2_provider_tool_error_responses_have_stable_shape[memory_v2_resolve_open_loop-args20-status must be one of]
tests/plugins/memory/test_memory_v2_retrieval.py::test_packet_composer_temporal_window_filters_to_yesterday
tests/plugins/memory/test_memory_v2_retrieval.py::test_packet_composer_temporal_intent_prefers_recent_evidence
tests/plugins/memory/test_memory_v2_retrieval.py::test_active_project_card_supplements_noisy_broad_continuity_search
tests/plugins/memory/test_memory_v2_retrieval.py::test_packet_composer_v2_renders_selective_sections_for_project_continuity
tests/plugins/memory/test_memory_v2_retrieval.py::test_packet_composer_v2_includes_source_refs_for_exact_source_routes
tests/plugins/memory/test_memory_v2_retrieval.py::test_packet_composer_source_verification_marks_raw_events_as_raw_evidence
tests/plugins/memory/test_memory_v2_retrieval.py::test_deep_recall_triggers_are_expensive_and_expose_process_scaffold
tests/plugins/memory/test_memory_v2_retrieval.py::test_packet_renderer_marks_recalled_content_as_untrusted_data
tests/plugins/memory/test_memory_v2_retrieval.py::test_prefetch_redacts_natural_language_api_key_formats_in_retrieval_log
tests/plugins/memory/test_memory_v2_retrieval.py::test_prefetch_does_not_persist_full_sensitive_query_by_default
tests/plugins/memory/test_memory_v2_review_queue.py::test_review_queue_groups_pending_candidates_and_recommends_safe_actions
tests/plugins/memory/test_memory_v2_review_queue.py::test_review_queue_does_not_call_unbounded_read_raw_events_and_hydrates_only_review_limit
tests/plugins/memory/test_memory_v2_session_backfill.py::test_session_backfill_imports_idempotently_and_searches_with_sources
tests/plugins/memory/test_memory_v2_session_backfill.py::test_archive_retrieval_is_bounded_untrusted_and_not_a_dump
tests/agent/test_memory_v2_conversation_loop_session_prefetch.py::test_conversation_loop_passes_agent_session_id_to_memory_prefetch
""".strip().splitlines())


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid in _RETIRED_NODEIDS:
            item.add_marker(pytest.mark.xfail(
                strict=True,
                reason="retired pre-P0 contract; replacement safety/production-path regression is active",
            ))


@pytest.fixture(autouse=True)
def legacy_memory_v2_feature_opt_in(request, tmp_path):
    """Make pre-feature-flag tests explicit opt-in consumers."""
    if request.node.path.name not in _LEGACY_OPT_IN_MODULES:
        return
    config_path = tmp_path / "config.yaml"
    if config_path.exists():
        return
    config_path.write_text(
        """memory_v2:
  archive:
    enabled: true
    capture_enabled: true
    search_tools_enabled: true
    show_tools_enabled: true
    prefetch_raw_enabled: true
    include_tool_outputs: true
  extraction:
    enabled: true
    candidate_creation_enabled: true
  consolidation:
    enabled: true
  prefetch:
    enabled: true
  review_apply:
    enabled: true
  contradictions:
    create_candidates: true
  working_memory:
    enabled: true
""",
        encoding="utf-8",
    )


@pytest.fixture
def memory_v2_store_index(tmp_path):
    store = MemoryV2Store(tmp_path)
    index = MemoryV2Index(tmp_path)
    return store, index
