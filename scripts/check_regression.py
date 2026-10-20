from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "data" / "baselines" / "regression_baseline.json"
INPUT_DIR = PROJECT_ROOT / "data" / "input"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
CONFIG_PATH = PROJECT_ROOT / "config.low-memory.yaml"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ACCURACY_TOLERANCE = 0.001


def compare_evaluations(
    baseline: dict[str, Any],
    current: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Return whether current evaluation meets the regression guard."""
    if current is None:
        return False, ["missing current evaluation"]

    failures = []
    if current.get("matched", -1) < baseline["matched"]:
        failures.append("matched decreased")
    if current.get("unexpected", float("inf")) > baseline["unexpected"]:
        failures.append("unexpected increased")
    if current.get("accuracy", float("-inf")) < (
        baseline["accuracy"] - ACCURACY_TOLERANCE
    ):
        failures.append("accuracy dropped beyond tolerance")
    return not failures, failures


def _run_plans() -> bool:
    """Run each available plan with its matching ground truth, if present."""
    inputs = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    success = True
    for image_path in inputs:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run.py"),
            str(image_path),
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(OUTPUT_DIR),
        ]
        ground_truth = GROUND_TRUTH_DIR / f"{image_path.stem}_ground_truth.json"
        if ground_truth.exists():
            command.extend(["--ground-truth", str(ground_truth)])
        print(f"\nRunning {image_path.name}")
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            print(f"Run failed for {image_path.name}: exit {completed.returncode}")
            success = False
    return success


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None when the file is missing or invalid."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    """Compare current plan evaluations against the checked-in baseline."""
    parser = argparse.ArgumentParser(
        description="Check saved plan evaluations against regression baselines."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run each plan first using config.low-memory.yaml and matching ground truth.",
    )
    args = parser.parse_args()

    baseline_data = _load_json(BASELINE_PATH)
    if baseline_data is None or not isinstance(baseline_data.get("plans"), dict):
        print(f"Invalid or missing baseline file: {BASELINE_PATH}")
        return 1

    runs_succeeded = True
    if args.run:
        runs_succeeded = _run_plans()

    failed = not runs_succeeded
    print(
        f"{'Plan':<8} {'Baseline (matched/missed/unexpected; accuracy)':<47} "
        f"{'Now (matched/missed/unexpected; accuracy)':<45} Status"
    )
    for plan, baseline in baseline_data["plans"].items():
        result = _load_json(OUTPUT_DIR / f"{plan}_result.json")
        evaluation = result.get("evaluation") if result else None
        passed, reasons = compare_evaluations(baseline, evaluation)
        failed = failed or not passed

        baseline_text = (
            f"{baseline['matched']}/{baseline['missed']}/{baseline['unexpected']}; "
            f"{baseline['accuracy']:.4f}"
        )
        if evaluation is None:
            current_text = "missing evaluation"
        else:
            current_text = (
                f"{evaluation.get('matched', '-')}/"
                f"{evaluation.get('missed', '-')}/"
                f"{evaluation.get('unexpected', '-')}; "
                f"{evaluation.get('accuracy', float('nan')):.4f}"
            )
        status = "PASS" if passed else f"FAIL ({', '.join(reasons)})"
        print(f"{plan:<8} {baseline_text:<47} {current_text:<45} {status}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
