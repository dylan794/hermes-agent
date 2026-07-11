from __future__ import annotations

import inspect

from agent import conversation_loop, turn_context


def test_conversation_loop_uses_turn_context_prefetch_without_spoofable_session_argument() -> None:
    context_source = inspect.getsource(turn_context.build_turn_context)
    loop_source = inspect.getsource(conversation_loop.run_conversation)

    assert "prefetch_all(_query)" in context_source
    assert "prefetch_all(_query, session_id=" not in context_source
    assert "build_memory_context_block(_ext_prefetch_cache)" in loop_source
    assert "api_msg[\"content\"] = _base +" in loop_source
    assert "effective_system = (effective_system + \"\\n\\n\" + _ext_prefetch_cache)" not in loop_source
