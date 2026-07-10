from __future__ import annotations

from dataclasses import replace

from plugins.memory.memory_v2.frontier_workflows import (
    FRONTIER_WORKFLOW_TEMPLATES,
    FrontierWorkflowTemplate,
    UltraWorkflowActivation,
    get_frontier_workflow_template,
    render_all_frontier_workflow_markdown,
    render_frontier_workflow_markdown,
    resolve_ultra_workflow,
)


EXPECTED_WORKFLOWS = {
    "repo_audit",
    "pr_triage",
    "memory_v2_red_team",
    "implementation_review_verification",
}


def test_frontier_workflow_templates_cover_requested_workflows():
    assert EXPECTED_WORKFLOWS <= set(FRONTIER_WORKFLOW_TEMPLATES)

    for workflow_id in EXPECTED_WORKFLOWS:
        template = FRONTIER_WORKFLOW_TEMPLATES[workflow_id]
        assert isinstance(template, FrontierWorkflowTemplate)
        assert template.id == workflow_id
        assert template.goal
        assert template.when_to_use
        assert template.stages
        assert template.final_artifacts
        assert template.abort_conditions


def test_frontier_workflow_stages_have_slots_gates_and_evidence_outputs():
    from hermes_cli.config import DEFAULT_CONFIG

    model_routing = DEFAULT_CONFIG["model_routing"]
    allowed_slots = set(model_routing["slots"])
    allowed_policies = set(model_routing["policies"])
    gate_types = {"preflight", "revision", "escalation", "abort"}

    for template in FRONTIER_WORKFLOW_TEMPLATES.values():
        assert template.model_policy in allowed_policies
        assert template.max_reasoning_effort in {"low", "medium", "high"}
        assert template.allow_ultra is False
        for stage in template.stages:
            assert stage.slot in allowed_slots
            assert stage.gate in gate_types
            assert stage.inputs
            assert stage.outputs
            assert stage.evidence_required
            assert stage.forbidden_actions


def test_memory_v2_red_team_workflow_is_privacy_and_source_grounding_focused():
    template = get_frontier_workflow_template("memory_v2_red_team")
    text = render_frontier_workflow_markdown(template).lower()

    assert "raw private text" in text
    assert "source refs" in text
    assert "instruction" in text
    assert "privacy" in text
    assert "memory_curator" in text or "memory curator" in text
    assert "tests/plugins/memory" in text


def test_implementation_workflow_enforces_builder_reviewer_verification_separation():
    template = get_frontier_workflow_template("implementation_review_verification")
    slots = [stage.slot for stage in template.stages]
    names = [stage.name.lower() for stage in template.stages]

    assert "implementation_worker" in slots
    assert "reviewer" in slots
    assert any("verify" in name or "verification" in name for name in names)
    assert any(stage.gate == "revision" for stage in template.stages)
    assert any("do not trust builder summaries" in " ".join(stage.forbidden_actions).lower() for stage in template.stages)


def test_pr_triage_workflow_is_read_only_until_explicit_gate():
    template = get_frontier_workflow_template("pr_triage")
    text = render_frontier_workflow_markdown(template).lower()

    assert "read-only" in text
    assert "do not push" in text
    assert "do not merge" in text
    assert "ci" in text
    assert "diff" in text


def test_repo_audit_workflow_uses_fanout_then_synthesis():
    template = get_frontier_workflow_template("repo_audit")
    stage_names = [stage.name.lower() for stage in template.stages]

    assert any("map" in name or "inventory" in name for name in stage_names)
    assert any("fanout" in name or "parallel" in name for name in stage_names)
    assert any("synth" in name for name in stage_names)
    assert any(stage.slot == "bulk_context_worker" for stage in template.stages)
    assert any(stage.slot == "orchestrator" for stage in template.stages)


def test_render_frontier_workflow_markdown_contains_machine_readable_summary():
    template = get_frontier_workflow_template("repo_audit")
    rendered = render_frontier_workflow_markdown(template)

    assert rendered.startswith("## Repo Audit Workflow")
    assert "```yaml" in rendered
    assert "workflow_id: repo_audit" in rendered
    assert "### Stages" in rendered
    assert "### Final artifacts" in rendered


def test_frontier_workflow_templates_doc_matches_generated_renderer():
    from pathlib import Path

    doc = Path("docs/frontier-workflow-templates.md").read_text(encoding="utf-8")

    assert doc == render_all_frontier_workflow_markdown()
    assert "# Frontier Workflow Templates" in doc
    assert "## Repo Audit Workflow" in doc
    assert "## PR Triage Workflow" in doc
    assert "## Memory v2 Red-Team Workflow" in doc
    assert "## Implementation + Review + Verification Workflow" in doc


def test_unknown_frontier_workflow_template_raises_key_error():
    try:
        get_frontier_workflow_template("missing")
    except KeyError as exc:
        assert "unknown frontier workflow template" in str(exc)
    else:
        raise AssertionError("missing template lookup should fail")


def _ultra_ready_inputs():
    from hermes_cli.config import DEFAULT_CONFIG

    routing = {
        **DEFAULT_CONFIG["model_routing"],
        "enabled": True,
        "allow_ultra_workflows_by_default": True,
    }
    template = replace(get_frontier_workflow_template("repo_audit"), allow_ultra=True)
    activation = UltraWorkflowActivation(
        explicit_opt_in=True,
        isolation_mode="sandbox",
        budget_usd=5.0,
        review_gate=True,
        evidence_gate=True,
    )
    return template, routing, activation


def test_ultra_workflow_activation_requires_every_gate():
    template, routing, activation = _ultra_ready_inputs()

    assert resolve_ultra_workflow(
        template, model_routing=routing, activation=activation
    ).enabled is True

    cases = (
        ({**routing, "enabled": False}, template, activation, "model_routing_disabled"),
        (
            {**routing, "allow_ultra_workflows_by_default": False},
            template,
            activation,
            "ultra_policy_not_opted_in",
        ),
        (
            routing,
            replace(template, allow_ultra=False),
            activation,
            "template_does_not_allow_ultra",
        ),
        (
            routing,
            template,
            replace(activation, explicit_opt_in=False),
            "run_not_explicitly_opted_in",
        ),
        (
            routing,
            template,
            replace(activation, isolation_mode=""),
            "isolation_required",
        ),
        (
            routing,
            template,
            replace(activation, budget_usd=0),
            "positive_budget_required",
        ),
        (
            routing,
            template,
            replace(activation, budget_usd=float("nan")),
            "positive_budget_required",
        ),
        (
            routing,
            template,
            replace(activation, budget_usd=float("inf")),
            "positive_budget_required",
        ),
        (
            routing,
            template,
            replace(activation, review_gate=False),
            "review_gate_required",
        ),
        (
            routing,
            template,
            replace(activation, evidence_gate=False),
            "evidence_gate_required",
        ),
        (
            {**routing, "require_evidence_for_done": False},
            template,
            activation,
            "evidence_gate_required",
        ),
    )
    for candidate_routing, candidate_template, candidate_activation, blocker in cases:
        resolution = resolve_ultra_workflow(
            candidate_template,
            model_routing=candidate_routing,
            activation=candidate_activation,
        )
        assert resolution.enabled is False
        assert blocker in resolution.blockers


def test_ultra_workflow_activation_is_provider_agnostic_and_fail_closed():
    template, routing, activation = _ultra_ready_inputs()
    routing_without_slots = {
        key: value for key, value in routing.items() if key not in {"slots", "fallback_slot"}
    }

    resolution = resolve_ultra_workflow(
        template,
        model_routing=routing_without_slots,
        activation=replace(activation, isolation_mode="worktree"),
    )
    assert resolution.enabled is True

    invalid = resolve_ultra_workflow(
        template,
        model_routing=routing_without_slots,
        activation=replace(activation, isolation_mode="host", budget_usd="unbounded"),
    )
    assert invalid.enabled is False
    assert set(invalid.blockers) >= {"isolation_required", "positive_budget_required"}
