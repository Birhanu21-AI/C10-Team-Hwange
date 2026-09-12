"""Access to configs/data.yaml.

The config holds research decisions, not tuning knobs. Loading it here keeps the
policy out of the loaders and lets every artefact record which policy produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from src import paths

DEFAULT_CONFIG_PATH = paths.CONFIGS_DIR / "data.yaml"


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load the data policy configuration."""
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def policy_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    """The subset of config that changes label semantics, stored with outputs."""
    return {
        "hatexplain.offensive_maps_to": config["hatexplain"]["offensive_maps_to"],
        "hatexplain.tie_posts": config["hatexplain"]["tie_posts"],
        "toxigen.threshold": config["toxigen"]["threshold"],
        "ubuntu.role": config["ubuntu"]["role"],
        "afrihate.enabled": config["afrihate"]["enabled"],
        "seed": config["seed"],
    }
