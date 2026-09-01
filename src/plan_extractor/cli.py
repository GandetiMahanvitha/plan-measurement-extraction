from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import extract_plan


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    """command-line argument parser for plan extraction.

    Centralizing the options keeps file, output, config, and evaluation inputs consistent.
    """
    parser = argparse.ArgumentParser(
        description="Extract architectural plan measurements into structured JSON."
    )
    parser.add_argument("input", help="Input plan image path.")
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory for generated JSON. Default: data/output",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration file.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Optional ground-truth JSON used for evaluation.",
    )
    return parser


def main() -> int:
    """
    This is the CLI entry point that validates input types and reports failures clearly.
    """
    args = build_parser().parse_args()
    input_path = Path(args.input)

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"Unsupported V1 input type: {input_path.suffix}. "
            "Use PNG/JPG/JPEG/TIFF/BMP. PDF/CAD are planned adapters."
        )
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_config(args.config)
        result = extract_plan(
            input_path,
            config,
            ground_truth_path=args.ground_truth,
        )
    except Exception as exc:
        print(f"Extraction failed: {exc}")
        return 1

    output_path = output_dir / f"{input_path.stem}_result.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    print(f"Status: {result.status}")
    print(f"Spaces detected: {result.summary.spaces_detected}")
    print(f"Measurements detected: {result.summary.measurements_detected}")
    print(f"Review required: {result.summary.review_required}")
    if result.evaluation:
        print(
            f"Evaluation: precision={result.evaluation.precision:.3f}, "
            f"recall={result.evaluation.recall:.3f}, "
            f"F1={result.evaluation.f1:.3f}"
        )
    print(f"Result: {output_path}")
    return 0
