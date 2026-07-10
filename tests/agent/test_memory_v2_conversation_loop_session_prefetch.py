from __future__ import annotations

import inspect

from agent import conversation_loop


def test_conversation_loop_passes_agent_session_id_to_memory_prefetch() -> None:
    source = inspect.getsource(conversation_loop.run_conversation)

    assert "_session_id = str(getattr(agent, \"session_id\", \"\") or \"\")" in source
    assert "prefetch_all(_query, session_id=_session_id)" in source
    assert "build_memory_context_block(_ext_prefetch_cache)" in source
    assert "api_msg[\"content\"] = _base +" in source
    assert "effective_system = (effective_system + \"\\n\\n\" + _ext_prefetch_cache)" not in source
