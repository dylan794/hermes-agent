from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="scripts/run_tests.sh is a POSIX shell runner",
)


def _make_fake_venv(
    venv: Path,
    *,
    label: str,
    pytest_available: bool,
    trace: Path,
) -> None:
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").touch()

    python = bin_dir / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"trace={shlex.quote(str(trace))}\n"
        "if [ \"${1:-}\" = \"-m\" ] "
        "&& [ \"${2:-}\" = \"pytest\" ] "
        "&& [ \"${3:-}\" = \"--version\" ]; then\n"
        f"  echo 'PREFLIGHT:{label}' >> \"$trace\"\n"
        f"  exit {0 if pytest_available else 1}\n"
        "fi\n"
        "if [ \"${1:-}\" = \"-m\" ] && [ \"${2:-}\" = \"compileall\" ]; then\n"
        f"  echo 'COMPILE:{label}' >> \"$trace\"\n"
        "  exit 0\n"
        "fi\n"
        f"echo 'RUN:{label}' >> \"$trace\"\n"
        f"echo 'SELECTED:{label}'\n"
    )
    python.chmod(0o755)


@pytest.mark.parametrize("fallback", ["venv", "shared"])
def test_runner_skips_unusable_dot_venv(
    tmp_path: Path,
    fallback: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_runner = repo_root / "scripts" / "run_tests.sh"

    fake_repo = tmp_path / "repo"
    fake_scripts = fake_repo / "scripts"
    fake_scripts.mkdir(parents=True)
    runner = fake_scripts / "run_tests.sh"
    runner.symlink_to(source_runner)
    (fake_scripts / "run_tests_parallel.py").touch()

    home = tmp_path / "home"
    home.mkdir()
    trace = tmp_path / "python-trace.txt"

    _make_fake_venv(
        fake_repo / ".venv",
        label="dot-venv",
        pytest_available=False,
        trace=trace,
    )

    if fallback == "venv":
        selected_venv = fake_repo / "venv"
    else:
        _make_fake_venv(
            fake_repo / "venv",
            label="venv",
            pytest_available=False,
            trace=trace,
        )
        selected_venv = home / ".hermes" / "hermes-agent" / "venv"

    _make_fake_venv(
        selected_venv,
        label=fallback,
        pytest_available=True,
        trace=trace,
    )

    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("HERMES_PYTHON", None)
    proc = subprocess.run(
        [str(runner), "tests/sentinel.py"],
        cwd=fake_repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert f"SELECTED:{fallback}" in proc.stdout
    assert "skipping unusable test virtualenv" in proc.stderr

    events = trace.read_text().splitlines()
    assert "PREFLIGHT:dot-venv" in events
    assert "RUN:dot-venv" not in events
    assert f"PREFLIGHT:{fallback}" in events
    assert f"COMPILE:{fallback}" in events
    assert f"RUN:{fallback}" in events
