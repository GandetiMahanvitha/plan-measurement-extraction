# scripts/run_batch.py
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path("data/input")
GROUND_TRUTH_DIR = Path("data/ground_truth")


def main():
    """Process every supported plan image in the input folder with the low-memory configuration."""
    images = sorted(INPUT_DIR.glob("*.png")) + sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.jpeg"))
    for image_path in images:
        stem = image_path.stem
        gt_path = GROUND_TRUTH_DIR / f"{stem}_ground_truth.json"
        #cmd = [sys.executable, "run.py", str(image_path)]
        cmd = [sys.executable, "run.py", str(image_path), "--config", "config.low-memory.yaml"]
        if gt_path.exists():
            cmd += ["--ground-truth", str(gt_path)]
        else:
            print(f"[!] No ground truth found for {image_path.name}, running without validation.")
        print(f"\n=== Processing {image_path.name} ===")
        subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()