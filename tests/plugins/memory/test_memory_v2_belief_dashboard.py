from plugins.memory.memory_v2.belief_dashboard import build_belief_update_dashboard
from plugins.memory.memory_v2.schemas import CandidateMemory, GateDecision, MemoryItem, MemoryStatus, MemoryType, SourceRef


def test_belief_update_dashboard_flags_uncertain_stale_conflicting_expiring_source_weak_items():
    now = "2026-06-27T00:00:00Z"
    items = [
        MemoryItem(id="low_conf", type=MemoryType.BELIEF, subject="Alex", value="likes verbose answers", confidence=0.35, source_refs=["src_old"], updated_at="2026-06-01T00:00:00Z"),
        MemoryItem(id="stale", type=MemoryType.ENVIRONMENT, subject="Hermes path", value="/old/path", confidence=0.8, source_refs=["src_old"], updated_at="2025-01-01T00:00:00Z"),
        MemoryItem(id="expiring", type=MemoryType.FACT, subject="school schedule", value="freshman year", confidence=0.8, source_refs=["src_old"], expires_at="2026-06-28T00:00:00Z"),
        MemoryItem(id="weak", type=MemoryType.FACT, subject="QQQ", value="strategy goal", confidence=0.9, source_refs=[], updated_at="2026-06-20T00:00:00Z"),
        MemoryItem(id="active_pref", type=MemoryType.PREFERENCE, subject="Alex response style", value="concise", confidence=0.9, source_refs=["src_new"], updated_at="2026-06-20T00:00:00Z"),
        MemoryItem(id="uncertain_pref", type=MemoryType.PREFERENCE, subject="Alex response style", value="long and formal", confidence=0.6, status=MemoryStatus.UNCERTAIN, source_refs=["src_old"], updated_at="2026-06-22T00:00:00Z"),
    ]
    candidates = [
        CandidateMemory(id="cand_low", type="belief", claim="Alex might prefer formal answers.", confidence=0.4, source_refs=["src_old"]),
        CandidateMemory(id="cand_conflict", type="preference", claim="Alex response style is detailed and formal.", confidence=0.8, source_refs=["src_new"], gate_decision=GateDecision.PENDING),
    ]
    sources = [
        SourceRef(id="src_old", type="message", uri="raw_event:src_old", observed_at="2025-01-01T00:00:00Z"),
        SourceRef(id="src_new", type="message", uri="raw_event:src_new", observed_at="2026-06-20T00:00:00Z"),
    ]

    dashboard = build_belief_update_dashboard(items=items, candidates=candidates, sources=sources, now=now, stale_days=365, expiring_days=7)

    assert dashboard["status"] == "draft"
    assert dashboard["policy"] == "report_only_no_mutation"
    assert {row["id"] for row in dashboard["low_confidence"]} >= {"low_conf", "cand_low"}
    assert {row["id"] for row in dashboard["stale"]} >= {"stale"}
    assert {row["id"] for row in dashboard["expiring"]} >= {"expiring"}
    assert {row["id"] for row in dashboard["source_weak"]} >= {"weak"}
    conflict_pairs = {(row["left_id"], row["right_id"]) for row in dashboard["conflicts"]}
    assert ("active_pref", "uncertain_pref") in conflict_pairs or ("uncertain_pref", "active_pref") in conflict_pairs
    assert dashboard["summary"]["suggested_actions"]
    assert all(action["mutation"] == "none" for action in dashboard["suggested_actions"])


def test_belief_dashboard_uses_hash_fingerprints_and_bounded_sections_without_raw_claim_fragments():
    private_phrase = "PRIVATE_BELIEF_RAW_FRAGMENT_SHOULD_NOT_LEAK"
    candidates = [
        CandidateMemory(
            id=f"cand_bulk_{idx:03d}",
            type="preference",
            claim=f"Alex response style is {private_phrase} variant {idx}.",
            confidence=0.4,
            source_refs=[f"src_{idx}"],
        )
        for idx in range(80)
    ]

    dashboard = build_belief_update_dashboard(candidates=candidates, now="2026-06-27T00:00:00Z")
    dashboard_json = __import__("json").dumps(dashboard)

    assert private_phrase not in dashboard_json
    assert all(private_phrase.lower() not in row["subject_key"] for row in dashboard["low_confidence"])
    assert all(row["subject_key"].startswith("subject:") for row in dashboard["low_confidence"])
    assert all(row["value_fingerprint"].startswith("sha256:") for row in dashboard["low_confidence"])
    assert len(dashboard["low_confidence"]) <= 50
    assert len(dashboard["conflicts"]) <= 50
    assert len(dashboard["suggested_actions"]) <= 50


def test_belief_dashboard_tolerates_bad_numeric_values():
    class WeirdRecord:
        id = "weird"
        type = "belief"
        subject = "Alex"
        value = "safe value"
        confidence = "not-a-float"
        importance = object()
        source_refs = []
        updated_at = "not-a-date"

    dashboard = build_belief_update_dashboard(items=[WeirdRecord()], now="2026-06-27T00:00:00Z")

    assert dashboard["summary"]["records_considered"] == 1
    assert dashboard["low_confidence"][0]["confidence"] == 0.0
