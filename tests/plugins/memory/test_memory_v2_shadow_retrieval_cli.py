from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import memory_v2_shadow_retrieval as shadow_cli


REPO_ROOT = Path(__file__).parents[3]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "scripts/memory_v2_shadow_retrieval.py", *args],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _request() -> dict:
    return {
        "query": "What did we previously decide for the alpha migration?",
        "raw_events": [
            {
                "id": "event-1",
                "type": "turn",
                "user_content": "Decision: use SQLite FTS for alpha migration.",
                "observed_at": "2026-05-01T10:00:00Z",
                "project_id": "alpha",
                "workstream_id": "migration",
                "profile_id": "profile-a",
                "tenant_id": "tenant-a",
            }
        ],
        "profile_id": "profile-a",
        "tenant_id": "tenant-a",
        "evidence_cutoff": "2026-06-01T00:00:00Z",
        "context": {
            "has_current_context": False,
            "gap_days": 30,
            "project_id": "alpha",
            "workstream_id": "migration",
        },
    }


@pytest.fixture
def external_temp_dir():
    root = (
        Path(os.environ["LOCALAPPDATA"]) / "Temp"
        if os.name == "nt"
        else Path("/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="memory-v2-shadow-cli-test-",
        dir=root,
    ) as directory:
        yield Path(directory)


def test_cli_requires_explicit_authorization_and_refuses_overwrite(
    external_temp_dir,
):
    request = external_temp_dir / "request.json"
    output = external_temp_dir / "result.json"
    request.write_text(json.dumps(_request()), encoding="utf-8")

    unauthorized = _run("--input", str(request), "--output", str(output))
    assert unauthorized.returncode == 2
    assert "authorize-private-shadow" in unauthorized.stderr
    assert not output.exists()

    authorized = _run(
        "--input",
        str(request),
        "--output",
        str(output),
        "--authorize-private-shadow",
    )
    assert authorized.returncode == 0, authorized.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    receipt = json.loads(authorized.stdout)
    assert payload["private_artifact"] is True
    assert payload["shadow_only"] is True
    assert payload["read_only"] is True
    assert payload["mutation_authority"] == "none"
    assert receipt["private_artifact"] is True
    assert "citations" not in authorized.stdout
    assert "SQLite FTS" not in authorized.stdout

    overwrite = _run(
        "--input",
        str(request),
        "--output",
        str(output),
        "--authorize-private-shadow",
    )
    assert overwrite.returncode == 2
    assert "overwrite" in overwrite.stderr


def test_cli_refuses_repo_local_private_paths(tmp_path):
    repo_request = REPO_ROOT / ".memory-v2-shadow-test-request.json"
    output = tmp_path / "result.json"
    try:
        repo_request.write_text(json.dumps(_request()), encoding="utf-8")
        completed = _run(
            "--input",
            str(repo_request),
            "--output",
            str(output),
            "--authorize-private-shadow",
        )
        assert completed.returncode == 2
        assert "outside the repository" in completed.stderr
        assert not output.exists()
    finally:
        repo_request.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "raw_input, error_fragment",
    [
        (
            '{"query":"first","query":"second"}',
            "duplicate JSON key",
        ),
        (
            '{"context":{"gap_days":NaN}}',
            "non-finite JSON value",
        ),
        (
            '{"context":{"gap_days":1e999}}',
            "non-finite JSON number",
        ),
    ],
)
def test_cli_rejects_ambiguous_or_nonfinite_json(
    external_temp_dir,
    raw_input,
    error_fragment,
):
    request = external_temp_dir / "request.json"
    output = external_temp_dir / "result.json"
    request.write_text(raw_input, encoding="utf-8")

    completed = _run(
        "--input",
        str(request),
        "--output",
        str(output),
        "--authorize-private-shadow",
    )

    assert completed.returncode == 2
    assert error_fragment in completed.stderr
    assert not output.exists()


def test_cli_rejects_excessive_json_nesting(external_temp_dir):
    request = external_temp_dir / "request.json"
    output = external_temp_dir / "result.json"
    request.write_text('{"nested":' * 70 + "null" + "}" * 70, encoding="utf-8")

    completed = _run(
        "--input",
        str(request),
        "--output",
        str(output),
        "--authorize-private-shadow",
    )

    assert completed.returncode == 2
    assert "nesting-depth bound" in completed.stderr
    assert not output.exists()


def test_cli_rejects_oversized_input_without_publishing(external_temp_dir):
    request = external_temp_dir / "request.json"
    output = external_temp_dir / "result.json"
    with request.open("wb") as handle:
        handle.seek(10_000_000)
        handle.write(b"}")

    completed = _run(
        "--input",
        str(request),
        "--output",
        str(output),
        "--authorize-private-shadow",
    )

    assert completed.returncode == 2
    assert "byte bound" in completed.stderr
    assert not output.exists()


def test_output_publication_is_atomic_and_never_clobbers(external_temp_dir):
    output = external_temp_dir / "result.json"
    barrier = threading.Barrier(2)

    def publish(marker: str) -> tuple[str, bool]:
        barrier.wait()
        try:
            shadow_cli._atomic_write_json(output, {"marker": marker})
        except ValueError as exc:
            assert "overwrite" in str(exc)
            return marker, False
        return marker, True

    with ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(publish, ("first", "second")))

    winners = [marker for marker, published in attempts if published]
    assert len(winners) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "marker": winners[0]
    }


def test_output_writer_rejects_nonfinite_results_without_partial_publication(
    external_temp_dir,
):
    output = external_temp_dir / "result.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        shadow_cli._atomic_write_json(output, {"score": float("nan")})

    assert not output.exists()
