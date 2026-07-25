from __future__ import annotations

import pytest

from plugins.memory.memory_v2.shadow_intent import ShadowMemoryNeedRouter


def test_disabled_router_never_requests_memory() -> None:
    decision = ShadowMemoryNeedRouter().route(
        "What did we decide last month?",
        {"has_current_context": False, "gap_days": 30},
    )

    assert decision.decision == "none"
    assert decision.reason == "shadow_router_disabled"


def test_immediate_context_followup_does_not_request_long_term_memory() -> None:
    router = ShadowMemoryNeedRouter(enabled=True)

    for query in (
        "Please fix the error that popped up after your message.",
        "Complete those steps.",
        "Can you explain this more?",
    ):
        decision = router.route(
            query,
            {"has_current_context": True, "gap_days": 0.0},
        )
        assert decision.decision == "none"
        assert decision.reason == "current_context_sufficient"


def test_explicit_history_and_long_gap_resumption_request_memory() -> None:
    router = ShadowMemoryNeedRouter(enabled=True)

    history = router.route(
        "Why did we choose the old indexing design last month?",
        {"has_current_context": False, "gap_days": 30},
    )
    resumed = router.route(
        "Resume the Memory v2 retrieval work from where we left off.",
        {
            "has_current_context": False,
            "gap_days": 7,
            "workstream_ids": ["workstream:memory-v2-retrieval"],
        },
    )

    assert history.decision == "needed"
    assert history.temporal_mode == "history"
    assert resumed.decision == "needed"
    assert resumed.temporal_mode == "current"


def test_acknowledgement_and_simple_question_abstain() -> None:
    router = ShadowMemoryNeedRouter(enabled=True)

    assert router.route("Thanks!", {}).decision == "none"
    assert router.route("What is 7 + 8?", {}).decision == "none"


def test_ambiguous_generic_query_uses_bounded_uncertain_fallback() -> None:
    decision = ShadowMemoryNeedRouter(enabled=True).route(
        "What should we do next?",
        {"has_current_context": False, "gap_days": 4},
    )

    assert decision.decision == "uncertain"
    assert decision.search_limit == 5
    assert decision.allow_unknown_workstream is True
    assert decision.confidence < 0.5


def test_current_and_history_intent_are_explicit() -> None:
    router = ShadowMemoryNeedRouter(enabled=True)

    current = router.route(
        "What is the current status of Memory v2?",
        {"has_current_context": False, "gap_days": 3},
    )
    history = router.route(
        "What was the previous status before it changed?",
        {"has_current_context": False, "gap_days": 3},
    )

    assert current.temporal_mode == "current"
    assert history.temporal_mode == "history"


def test_invalid_context_fails_closed() -> None:
    router = ShadowMemoryNeedRouter(enabled=True)

    decision = router.route(
        "Resume the project.",
        {"has_current_context": "yes", "gap_days": -1},
    )

    assert decision.decision == "none"
    assert decision.reason == "invalid_context"


@pytest.mark.parametrize(
    "gap_days",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        10**1000,
        1_000_000,
    ],
)
def test_nonfinite_or_implausible_gap_fails_closed(gap_days) -> None:
    decision = ShadowMemoryNeedRouter(enabled=True).route(
        "Resume the project.",
        {"has_current_context": False, "gap_days": gap_days},
    )

    assert decision.decision == "none"
    assert decision.reason == "invalid_context"


def test_large_but_plausible_gap_remains_valid() -> None:
    decision = ShadowMemoryNeedRouter(enabled=True).route(
        "Resume the project.",
        {"has_current_context": False, "gap_days": 3650},
    )

    assert decision.decision == "needed"


def test_before_and_old_do_not_override_an_explicit_current_target() -> None:
    router = ShadowMemoryNeedRouter(enabled=True)

    before = router.route(
        "Before more automation, what is the current plan?",
        {"has_current_context": False, "gap_days": 30},
    )
    old = router.route(
        "Is the old path still current?",
        {"has_current_context": False, "gap_days": 30},
    )

    assert before.temporal_mode == "current"
    assert old.temporal_mode == "current"
