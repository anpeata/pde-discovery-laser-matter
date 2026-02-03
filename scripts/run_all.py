"""Run the mldm-project pipelines end-to-end.

This is a lightweight orchestration script so you can reproduce the
latest outputs without opening a notebook.

Usage (from repo root or anywhere):
  python m2/mldm-project/scripts/run_all.py
  python m2/mldm-project/scripts/run_all.py --skip-heavy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    # scripts/ sits under <project_root>/scripts/
    return Path(__file__).resolve().parents[1]


def run_script(root: Path, script_name: str) -> None:
    script_path = root / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    print(f"\n=== Running: {script_path.name} ===")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)


def list_latest_pngs(root: Path, limit: int = 20) -> None:
    out_dir = root / "outputs" / "latest"
    if not out_dir.exists():
        print(f"No outputs directory found at: {out_dir}")
        return

    paths = sorted(
        (p for p in out_dir.rglob("*.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    print("\nMost recent PNG outputs:")
    for p in paths[:limit]:
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        print(f"- {p.relative_to(root)}  ({ts:%Y-%m-%d %H:%M:%S})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mldm-project pipelines")
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip longer-running scripts (registration, patch SINDy, transport)",
    )
    args = parser.parse_args()

    root = project_root()

    # Fast / always
    # Order matters: analyze_results writes best_model.json used by figures/slides.
    run_script(root, "analyze_results.py")
    run_script(root, "generate_presentation_figures_minimal.py")
    run_script(root, "create_simple_slide3.py")
    run_script(root, "create_method_slide.py")

    if not args.skip_heavy:
        run_script(root, "pde_discovery_improved_registration.py")
        run_script(root, "patch_based_sindy.py")
        run_script(root, "patch_based_pde_discovery.py")
        run_script(root, "transport_pde_discovery.py")

    list_latest_pngs(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
