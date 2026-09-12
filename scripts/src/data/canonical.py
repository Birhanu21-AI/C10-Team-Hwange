"""Phase 4: build the canonical dataset from all available sources.

Run:
    python -m src.data.canonical
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src import config as config_module
from src import paths
from src.data import loaders, schema

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_PATH = paths.PROCESSED_DIR / "canonical.parquet"
METADATA_PATH = paths.PROCESSED_DIR / "canonical_metadata.json"


def build_canonical(config: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """Load every configured source and concatenate into the canonical schema."""
    settings = config or config_module.load_config()
    paths.ensure_dirs()

    frames = []
    for name, loader in loaders.LOADERS.items():
        frame = loader(settings)
        if len(frame):
            frames.append(frame)

    if not frames:
        raise RuntimeError("no sources produced any rows")

    combined = schema.conform(pd.concat(frames, ignore_index=True))
    combined["is_synthetic"] = combined["is_synthetic"].astype("boolean")
    combined["label_binary"] = combined["label_binary"].astype("Int64")
    return combined


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Distribution summary of the canonical dataset."""
    return {
        "n_rows": int(len(frame)),
        "n_groups": int(frame["group_id"].nunique()),
        "by_source": frame["source"].value_counts().to_dict(),
        "by_language": frame["language"].value_counts().to_dict(),
        "by_label": {str(k): int(v) for k, v in frame["label_binary"].value_counts(dropna=False).items()},
        "by_synthetic": {str(k): int(v) for k, v in frame["is_synthetic"].value_counts(dropna=False).items()},
        "by_source_and_label": {
            source: {str(k): int(v) for k, v in group["label_binary"].value_counts(dropna=False).items()}
            for source, group in frame.groupby("source")
        },
        "unlabelled_rows": int(frame["label_binary"].isna().sum()),
    }


def main() -> None:
    settings = config_module.load_config()
    frame = build_canonical(settings)

    problems = schema.validate(frame)
    for problem in problems:
        logger.warning("schema: %s", problem)

    frame.to_parquet(CANONICAL_PATH, index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": config_module.policy_fingerprint(settings),
        "schema_problems": problems,
        "summary": summarize(frame),
    }
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    logger.info("wrote %s (%d rows)", CANONICAL_PATH, len(frame))
    logger.info("wrote %s", METADATA_PATH)
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()
