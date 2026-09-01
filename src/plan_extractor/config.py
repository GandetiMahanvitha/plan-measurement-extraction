from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

def load_config(path: str | Path | None = None) -> dict:
    """Load YAML configuration from the supplied path or repository default.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
