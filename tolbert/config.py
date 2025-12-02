import json
from pathlib import Path
from typing import Any, Dict


def load_tolbert_config(path: str) -> Dict[str, Any]:
    """
    Load a simple YAML or JSON config file into a Python dict.

    - If the extension is .yaml or .yml, this function requires PyYAML.
    - If the extension is .json, it uses the standard library.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
            ) from e

        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if suffix == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension: {suffix} (expected .yaml, .yml, or .json)")


