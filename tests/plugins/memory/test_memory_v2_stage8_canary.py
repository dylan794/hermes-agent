from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from plugins.memory.memory_v2.config import MemoryV2FeatureFlags, load_memory_v2_config


REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / "docs" / "memory-v2-stage-8-canary.md"
EXAMPLE = (
    REPO_ROOT
    / "plugins"
    / "memory"
    / "memory_v2"
    / "stage_8_canary_config.example.yaml"
)
SCRIPT = REPO_ROOT / "scripts" / "memory_v2_stage8_canary.py"


def _flatten(value: Any, prefix: str = "memory_v2") -> dict[str, bool]:
    flattened: dict[str, bool] = {}
    for field in fields(value):
        child = getattr(value, field.name)
        path = f"{prefix}.{field.name}"
        if is_dataclass(child):
            flattened.update(_flatten(child, path))
        else:
            assert isinstance(child, bool)
            flattened[path] = child
    return flattened


def _matrix() -> dict[str, tuple[bool, bool, bool]]:
    rows: dict[str, tuple[bool, bool, bool]] = {}
    pattern = re.compile(
        r"^\| `(?P<flag>memory_v2\.[^`]+)` \| `(?P<default>true|false)` "
        r"\| `(?P<canary>true|false)` \| `(?P<rollback>true|false)` \|"
    )
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            rows[match.group("flag")] = tuple(
                match.group(name) == "true"
                for name in ("default", "canary", "rollback")
            )
    return rows


def test_stage8_example_and_matrix_match_runtime_config(tmp_path) -> None:
    parsed = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    assert parsed["memory"]["provider"] == "memory_v2"
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(parsed, sort_keys=False), encoding="utf-8"
    )

    defaults = _flatten(MemoryV2FeatureFlags())
    canary = _flatten(load_memory_v2_config(tmp_path))
    matrix = _matrix()

    assert set(matrix) == set(defaults) == set(canary)
    assert {flag: values[0] for flag, values in matrix.items()} == defaults
    assert {flag: values[1] for flag, values in matrix.items()} == canary


def test_stage8_dangerous_paths_remain_disabled_in_parsed_example(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(
        EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    flags = load_memory_v2_config(tmp_path)

    assert flags.prefetch.enabled is True
    assert flags.archive.capture_enabled is True
    assert flags.extraction.enabled is True
    assert flags.archive.prefetch_raw_enabled is False
    assert flags.archive.include_tool_outputs is False
    assert flags.extraction.small_model_enabled is False
    assert flags.review_apply.enabled is False
    assert flags.auto_promote.enabled is False
    assert flags.contradictions.auto_supersede is False


def test_stage8_doc_names_example_command_go_no_go_and_rollback() -> None:
    text = DOC.read_text(encoding="utf-8")

    assert "stage_8_canary_config.example.yaml" in text
    assert "python scripts/memory_v2_stage8_canary.py" in text
    assert "Measurable go criteria" in text
    assert "No-go and rollback" in text
    assert "fresh_temporary_directory_only" in text
    assert "byte-for-byte identical" in text


def test_stage8_canary_is_deterministic_and_passes() -> None:
    def run() -> tuple[str, dict[str, Any]]:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        output = completed.stdout.strip()
        return output, json.loads(output)

    first_output, first = run()
    second_output, second = run()

    assert first_output == second_output
    assert first == second
    assert first["success"] is True
    assert first["go"] is True
    assert first["profile_scope"] == "fresh_temporary_directory_only"
    assert first["checks"]
    assert all(first["checks"].values())
    assert first["metrics"] == {
        "candidates": 1,
        "memory_items": 1,
        "pending_candidates": 0,
        "raw_events": 1,
        "review_actions": 1,
    }
