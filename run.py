from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plan_extractor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
