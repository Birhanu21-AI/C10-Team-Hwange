"""Phase 6: leakage-safe dataset splitting.

Rules, in order of precedence:
  1. A group_id is indivisible. Templates, near-duplicates and translation pairs
     all share a group and always land in the same split.
  2. Official splits are honoured where a source provides them. Where an official
     split would break rule 1, the whole group is reassigned to the split holding
     most of its rows, and the reassignment is reported rather than hidden.
  3. A source configured as `probe_only` never enters training. Ubuntu is held out
     entirely as a diagnostic set, because 33 templates cannot support training.

Run:
    python -m src.data.splits
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src import config as config_module
from src import paths
from src.data.canonical import CANONICAL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SPLIT_PATH = paths.PROCESSED_DIR / "canonical_split.parquet"
REPORT_PATH = paths.REPORTS_DIR / "phase6_splits" / "split_report.json"

DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _assign_groups_randomly(
    groups: list[str], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Assign whole groups to splits by the given ratios."""
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    assignment = {}
    for index, group in enumerate(shuffled):
        if index < n_train:
            assignment[group] = "train"
        elif index < n_train + n_val:
            assignment[group] = "val"
        else:
            assignment[group] = "test"
    return assignment


def _resolve_official_splits(frame: pd.DataFrame) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Map each group to one official split, reporting groups that spanned several."""
    assignment: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    labelled = frame[frame["official_split"].notna()]
    for group_id, group in labelled.groupby("group_id"):
        counts = group["official_split"].value_counts()
        winner = str(counts.index[0])
        assignment[group_id] = winner
        if len(counts) > 1:
            conflicts.append(
                {
                    "group_id": group_id,
                    "spanned": counts.to_dict(),
                    "assigned_to": winner,
                    "reason": "group indivisible; assigned to the split holding most of its rows",
                }
            )
    return assignment, conflicts


def build_splits(config: Optional[dict[str, Any]] = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a `split` column to the canonical dataset without breaking any group."""
    settings = config or config_module.load_config()
    seed = int(settings["seed"])
    frame = pd.read_parquet(CANONICAL_PATH)
    frame["split"] = pd.NA

    report: dict[str, Any] = {"seed": seed, "per_source": {}, "official_split_conflicts": []}

    for source, group in frame.groupby("source"):
        source_policy = settings.get(source, {})
        role = source_policy.get("role")

        if role == "probe_only":
            frame.loc[group.index, "split"] = "probe"
            report["per_source"][source] = {
                "strategy": "held out entirely as a probe set",
                "reason": source_policy.get("role_reason", "configured role=probe_only"),
                "n_rows": int(len(group)),
                "n_groups": int(group["group_id"].nunique()),
            }
            continue

        has_official = group["official_split"].notna().all()
        if has_official:
            assignment, conflicts = _resolve_official_splits(group)
            report["official_split_conflicts"].extend(conflicts)
            strategy = "official splits, made group-consistent"
        else:
            assignment = _assign_groups_randomly(
                sorted(group["group_id"].unique()), DEFAULT_RATIOS, seed
            )
            strategy = f"group-level random split {DEFAULT_RATIOS} (seed={seed})"

        frame.loc[group.index, "split"] = group["group_id"].map(assignment).values
        report["per_source"][source] = {
            "strategy": strategy,
            "n_rows": int(len(group)),
            "n_groups": int(group["group_id"].nunique()),
            "n_groups_reassigned": len([c for c in report["official_split_conflicts"]]),
        }

    unassigned = int(frame["split"].isna().sum())
    if unassigned:
        logger.warning("%d rows have no split assignment", unassigned)
    report["unassigned_rows"] = unassigned
    return frame, report


def verify_no_leakage(frame: pd.DataFrame) -> dict[str, Any]:
    """Confirm no group and no normalized text spans more than one split."""
    splits_per_group = frame.groupby("group_id")["split"].nunique()
    leaking_groups = splits_per_group[splits_per_group > 1]
    splits_per_text = frame.groupby(frame["text"].astype(str))["split"].nunique()
    leaking_texts = splits_per_text[splits_per_text > 1]
    return {
        "groups_spanning_splits": int(len(leaking_groups)),
        "texts_spanning_splits": int(len(leaking_texts)),
        "clean": bool(len(leaking_groups) == 0 and len(leaking_texts) == 0),
    }


def evaluation_views(frame: pd.DataFrame) -> dict[str, Any]:
    """The evaluation slices Phase 11-12 will report separately."""
    trainable = frame[frame["split"] == "train"]
    return {
        "in_domain": {
            "train": int(len(trainable)),
            "val": int((frame["split"] == "val").sum()),
            "test": int((frame["split"] == "test").sum()),
        },
        "cross_source": {
            source: int(len(group[group["split"] == "test"]))
            for source, group in frame.groupby("source")
        },
        "african_language": {
            language: int(len(group))
            for language, group in frame[frame["language"] != "en"].groupby("language")
        },
        "ubuntu_probe": int((frame["split"] == "probe").sum()),
    }


def main() -> None:
    frame, report = build_splits()
    report["leakage_check"] = verify_no_leakage(frame)
    report["evaluation_views"] = evaluation_views(frame)
    report["split_sizes"] = {str(k): int(v) for k, v in frame["split"].value_counts().items()}
    report["split_by_source"] = {
        f"{source}/{split}": int(len(g))
        for (source, split), g in frame.groupby(["source", "split"])
    }

    frame.to_parquet(SPLIT_PATH, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    logger.info("wrote %s", SPLIT_PATH)
    print(json.dumps(
        {k: report[k] for k in ["split_sizes", "split_by_source", "leakage_check", "evaluation_views"]},
        indent=2,
    ))


if __name__ == "__main__":
    main()
