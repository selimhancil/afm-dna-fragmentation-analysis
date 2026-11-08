"""Record the software environment of a run, so small numeric differences between
machines (e.g. 0.1-0.7 px in pair spacing) can be traced to library versions."""
from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "scipy", "scikit-image", "pandas", "pillow", "matplotlib")


def collect() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def compare(a: dict, b: dict) -> list[str]:
    """Human-readable list of differences between two environment records."""
    out = []
    for key in ("python", "machine"):
        if a.get(key) != b.get(key):
            out.append(f"{key}: {a.get(key)} vs {b.get(key)}")
    for name in sorted(set(a.get("packages", {})) | set(b.get("packages", {}))):
        va, vb = a.get("packages", {}).get(name), b.get("packages", {}).get(name)
        if va != vb:
            out.append(f"{name}: {va} vs {vb}")
    return out


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "outputs" / "environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    lines = ["# Run environment", "",
             f"- Python: {env['python']}", f"- System: {env['platform']} ({env['machine']})", ""]
    lines += [f"- {k}: {v}" for k, v in env["packages"].items()]
    lines += ["", "If library versions differ between machines, some intermediate numbers can "
              "move by very small (sub-pixel) amounts. Verdict counts should not be affected by this; "
              "to compare two machines, see the environment.json files."]
    (out_dir / "environment.md").write_text("\n".join(lines) + "\n")
    print(f"Written: {out_dir}")
    return 0
