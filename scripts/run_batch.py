from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plan_extractor.config import load_config
from plan_extractor.ocr_engine import EasyOCREngine
from plan_extractor.pipeline import extract_plan


INPUT_DIR = PROJECT_ROOT / "data" / "input"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
CONFIG_PATH = PROJECT_ROOT / "config.low-memory.yaml"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def main() -> int:
    """Process every plan with one configured, shared EasyOCR engine."""
    images = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        print(f"No supported plan images found in {INPUT_DIR}")
        return 1

    config = load_config(CONFIG_PATH)
    engine = EasyOCREngine(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = False

    for image_path in images:
        ground_truth_path = GROUND_TRUTH_DIR / f"{image_path.stem}_ground_truth.json"
        print(f"\n=== Processing {image_path.name} ===")
        try:
            result = extract_plan(
                image_path,
                config,
                ground_truth_path=(
                    ground_truth_path if ground_truth_path.exists() else None
                ),
                engine=engine,
            )
            output_path = OUTPUT_DIR / f"{image_path.stem}_result.json"
            output_path.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Status: {result.status}")
            print(f"Measurements detected: {result.summary.measurements_detected}")
            print(f"Result: {output_path}")
        except Exception as exc:
            failed = True
            print(f"Extraction failed for {image_path.name}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
