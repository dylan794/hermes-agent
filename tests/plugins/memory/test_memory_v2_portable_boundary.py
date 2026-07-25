from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "plugins" / "memory" / "memory_v2"
MANIFEST_PATH = PACKAGE_DIR / "portable_manifest.json"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_portable_manifest_is_complete_and_versioned():
    manifest = _manifest()

    assert manifest["schema_version"] == "memory-v2-portable-manifest/v1"
    assert manifest["status"] == "hybrid-development"
    assert manifest["adapter_entrypoint"] == "__init__.py"

    modules = manifest["portable_modules"]
    assert isinstance(modules, list)
    assert modules
    assert len(modules) == len(set(modules))
    for module_name in modules:
        assert isinstance(module_name, str)
        assert (PACKAGE_DIR / f"{module_name}.py").is_file()


def test_portable_modules_import_without_loading_hermes_runtime():
    probe = r"""
import importlib
import json
import pathlib
import sys
import types

package_dir = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads((package_dir / "portable_manifest.json").read_text(encoding="utf-8"))
before = set(sys.modules)

package_name = "_memory_v2_portable_probe"
package = types.ModuleType(package_name)
package.__path__ = [str(package_dir)]
package.__package__ = package_name
sys.modules[package_name] = package

for module_name in manifest["portable_modules"]:
    importlib.import_module(f"{package_name}.{module_name}")

loaded = set(sys.modules) - before
forbidden = tuple(manifest["forbidden_import_prefixes"])
offenders = sorted(
    name
    for name in loaded
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
if offenders:
    raise SystemExit("forbidden Hermes imports loaded: " + ", ".join(offenders))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(PACKAGE_DIR)],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
