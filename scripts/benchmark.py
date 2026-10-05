from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plan_extractor.config import load_config
from plan_extractor.ocr_engine import EasyOCREngine
from plan_extractor.pipeline import extract_plan


INPUT_DIR = PROJECT_ROOT / "data" / "input"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
RESULTS_PATH = PROJECT_ROOT / "data" / "benchmark_results.csv"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CSV_FIELDS = ["setting", "plan", "seconds", "correct", "incorrect", "missed"]
SETTING_IDS = ("A", "B", "C", "D")
PLAN2_BASELINE = {"correct": 4, "incorrect": 1, "missed": 1}


def _parse_list(value: str, allowed: tuple[str, ...], option: str) -> list[str]:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not entries:
        raise ValueError(f"{option} must contain at least one value.")
    invalid = [entry for entry in entries if entry not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported {option} value(s): {', '.join(invalid)}. "
            f"Choose from {', '.join(allowed)}."
        )
    return list(dict.fromkeys(entries))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR settings against plan ground truth."
    )
    parser.add_argument(
        "--plans",
        default="plan2",
        help='Comma-separated plan names or "all" (default: plan2).',
    )
    parser.add_argument(
        "--settings",
        default="A,B,C,D",
        help="Comma-separated setting IDs A, B, C, D (default: all four).",
    )
    return parser


def _setting_configs() -> dict[str, dict]:
    low_memory = load_config(PROJECT_ROOT / "config.low-memory.yaml")
    baseline = deepcopy(low_memory)
    baseline["ocr"]["general_passes"] = 3

    one_pass = deepcopy(low_memory)
    one_pass["ocr"]["general_passes"] = 1

    two_rotations = deepcopy(one_pass)
    two_rotations["ocr"]["rotation_info"] = [90, 270]

    standard_one_pass = load_config(PROJECT_ROOT / "config.yaml")
    standard_one_pass["ocr"]["general_passes"] = 1
    return {
        "A": baseline,
        "B": one_pass,
        "C": two_rotations,
        "D": standard_one_pass,
    }


def _read_csv_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0:
        return []
    with RESULTS_PATH.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                f"{RESULTS_PATH} must have CSV columns: {', '.join(CSV_FIELDS)}"
            )
        return list(reader)


def _baseline_by_plan(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    baselines = {}
    for row in rows:
        if row["setting"] == "A":
            baselines[row["plan"]] = {
                key: int(row[key]) for key in ("correct", "incorrect", "missed")
            }
    return baselines


def _comparison_status(
    row: dict[str, str],
    baselines: dict[str, dict[str, int]],
) -> str:
    baseline = baselines.get(row["plan"])
    if baseline is None and row["plan"].lower() == "plan2":
        baseline = PLAN2_BASELINE
    if baseline is None:
        return "NO A BASELINE"

    current = {key: int(row[key]) for key in ("correct", "incorrect", "missed")}
    return "SAME" if current == baseline else "DIFFERENT"


def _print_csv_table(rows: list[dict[str, str]]) -> None:
    print("\nBenchmark results (data/benchmark_results.csv):")
    print("| Setting | Plan | Seconds | Correct | Incorrect | Missed | vs A |")
    print("|---|---|---:|---:|---:|---:|---|")
    baselines = _baseline_by_plan(rows)
    for row in rows:
        print(
            f"| {row['setting']} | {row['plan']} | {float(row['seconds']):.2f} | "
            f"{row['correct']} | {row['incorrect']} | {row['missed']} | "
            f"{_comparison_status(row, baselines)} |"
        )


def main() -> int:
    args = _build_parser().parse_args()

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    image_by_plan = {path.stem.lower(): path for path in image_paths}

    plan_arg = args.plans.strip()
    if plan_arg.lower() == "all":
        plans = list(image_by_plan)
    else:
        try:
            plans = _parse_list(
                plan_arg,
                tuple(image_by_plan),
                "--plans",
            )
        except ValueError as exc:
            _build_parser().error(str(exc))
    try:
        settings = _parse_list(args.settings, SETTING_IDS, "--settings")
    except ValueError as exc:
        _build_parser().error(str(exc))

    for plan in plans:
        image_path = image_by_plan[plan]
        ground_truth_path = (
            GROUND_TRUTH_DIR / f"{image_path.stem}_ground_truth.json"
        )
        if not ground_truth_path.exists():
            raise FileNotFoundError(
                f"Benchmark requires ground truth for {image_path.name}: "
                f"{ground_truth_path}"
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_csv_rows()
    baselines = _baseline_by_plan(existing_rows)
    config_by_setting = _setting_configs()

    for setting in settings:
        config = config_by_setting[setting]
        print(f"\nPreparing EasyOCR engine for setting {setting}.", flush=True)
        engine = EasyOCREngine(config)

        for plan in plans:
            image_path = image_by_plan[plan]
            ground_truth_path = (
                GROUND_TRUTH_DIR / f"{image_path.stem}_ground_truth.json"
            )
            print(f"Starting setting {setting}, plan {plan}.", flush=True)
            started = perf_counter()
            result = extract_plan(
                image_path,
                config,
                ground_truth_path=ground_truth_path,
                engine=engine,
            )
            seconds = perf_counter() - started
            if result.evaluation is None:
                raise RuntimeError(
                    f"Evaluation was not generated for {image_path.name}."
                )

            evaluation = result.evaluation
            row = {
                "setting": setting,
                "plan": plan,
                "seconds": f"{seconds:.3f}",
                "correct": evaluation.matched,
                "incorrect": evaluation.incorrect_value + evaluation.incorrect_unit,
                "missed": evaluation.missed,
            }
            write_header = (
                not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
            )
            with RESULTS_PATH.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()

            string_row = {key: str(value) for key, value in row.items()}
            existing_rows.append(string_row)
            if setting == "A":
                baselines[plan] = {
                    key: int(row[key]) for key in ("correct", "incorrect", "missed")
                }
            print(f"Finished setting {setting}, plan {plan}: {seconds:.2f}s.", flush=True)

    _print_csv_table(_read_csv_rows())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
