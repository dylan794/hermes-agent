from plugins.memory.memory_v2.entity_graph import build_entity_graph_draft, extract_entity_links
from plugins.memory.memory_v2.schemas import CandidateMemory, MemoryItem, MemoryType, ProjectCard


def test_extract_entity_links_is_deterministic_and_does_not_leak_raw_claim_text():
    item = MemoryItem(
        id="mem_1",
        type=MemoryType.PROJECT_STATE,
        subject="Memory v2",
        predicate="uses",
        value="LoCoMo-style retrieval evals in Hermes",
        body="Memory v2 should improve stale-fact retrieval without LLM calls.",
        source_refs=["src_1"],
    )

    links = extract_entity_links(item)

    assert [link.bucket for link in links][:3] == ["memory_system", "memory_eval", "assistant_platform"]
    assert all(link.entity_id.startswith("entity:") for link in links)
    assert all(link.record_id.startswith("record:") for link in links)
    assert all("mem_1" not in link.record_id for link in links)
    assert all(link.source_refs == ("src_1",) for link in links)
    assert all("retrieval evals" not in link.to_dict().get("label", "").lower() for link in links)
    assert all("memory v2" not in link.to_dict().get("label", "").lower() for link in links)


def test_build_entity_graph_draft_links_memory_project_and_candidates_without_mutation():
    item = MemoryItem(id="mem_1", type="belief", subject="Hermes", value="Memory v2 needs source-grounded recall", source_refs=["src_1"])
    card = ProjectCard(id="project:memory-v2", name="Memory v2", current_state="Improve LoCoMo stale-fact retrieval", source_refs=["src_2"])
    candidate = CandidateMemory(id="cand_1", type="fact", claim="The user prefers QQQ research to be rigorous and source-grounded.", source_refs=["src_3"])

    draft = build_entity_graph_draft(memory_items=[item], project_cards=[card], candidates=[candidate])

    assert draft["status"] == "draft"
    assert draft["policy"] == "report_only_no_mutation"
    entity_buckets = {entity["bucket"] for entity in draft["entities"]}
    assert {"assistant_platform", "memory_system", "memory_eval", "market_index", "user_person"} <= entity_buckets
    graph_json = str(draft).lower()
    assert "entity:hermes" not in graph_json
    assert "entity:memory-v2" not in graph_json
    assert "record:project:memory-v2" not in graph_json
    assert any(edge["from"].startswith("entity:") and edge["to"].startswith("record:") for edge in draft["edges"])
    assert all("claim" not in edge and "body" not in edge and "raw_text" not in edge for edge in draft["edges"])
    assert draft["summary"]["entity_count"] == len(draft["entities"])



def test_entity_graph_caps_records_entities_edges_and_stringifies_list_fields():
    class WeirdRecord:
        id = "weird"
        type = "fact"
        subject = "Hermes Memory v2"
        value = "LoCoMo QQQ Nasdaq Qwen TTS the user"
        source_refs = [1, None]
        decisions = ["Decision text", 42]
        open_questions = [object()]
        next_actions = ["Next action"]
        related_entities = ["Memory v2", 99]

    records = [WeirdRecord() for _ in range(100)]

    draft = build_entity_graph_draft(memory_items=records, max_records=10, max_entities=5, max_edges=6)

    assert draft["summary"]["records_considered"] == 10
    assert len(draft["entities"]) <= 5
    assert len(draft["edges"]) <= 6
    assert all(all(isinstance(ref, str) for ref in edge["source_refs"]) for edge in draft["edges"])
