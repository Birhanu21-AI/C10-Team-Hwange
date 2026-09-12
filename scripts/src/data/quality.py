"""Phase 5: data quality and leakage control.

Nothing is removed automatically. Every issue is counted, described and written
to a report; a `removals` list records what a cleaning pass *would* drop and why,
leaving the decision to the researcher.

Run:
    python -m src.data.quality
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from src import paths
from src.data.canonical import CANONICAL_PATH
from src.data.profiling import normalize_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = paths.REPORTS_DIR / "phase5_quality" / "quality_report.json"


def check_empty_and_missing(frame: pd.DataFrame) -> dict[str, Any]:
    """Empty text and per-column missingness."""
    text = frame["text"].astype(str)
    return {
        "empty_text_rows": int((text.str.strip() == "").sum()),
        "null_text_rows": int(frame["text"].isna().sum()),
        "null_label_binary": int(frame["label_binary"].isna().sum()),
        "missing_by_column": {
            column: int(frame[column].isna().sum())
            for column in frame.columns
            if frame[column].isna().any()
        },
    }


def check_duplicates(frame: pd.DataFrame) -> dict[str, Any]:
    """Exact, normalized and cross-source duplicate text."""
    normalized = frame["text"].astype(str).map(normalize_text)
    exact_dupes = int(frame["text"].duplicated().sum())
    normalized_dupes = int(normalized.duplicated().sum())

    by_norm = frame.assign(_norm=normalized).groupby("_norm")["source"].nunique()
    cross_source = by_norm[by_norm > 1]

    return {
        "exact_duplicate_rows": exact_dupes,
        "normalized_duplicate_rows": normalized_dupes,
        "cross_source_duplicate_texts": int(len(cross_source)),
        "cross_source_examples": [str(t)[:120] for t in cross_source.index[:5]],
        "per_source": {
            source: {
                "n_rows": int(len(group)),
                "n_distinct_text": int(group["text"].nunique()),
                "repetition_factor": round(len(group) / max(group["text"].nunique(), 1), 2),
            }
            for source, group in frame.groupby("source")
        },
    }


def check_label_consistency(frame: pd.DataFrame) -> dict[str, Any]:
    """Groups whose rows disagree on label_binary — a group cannot be split by label."""
    labels_per_group = frame.groupby("group_id")["label_binary"].nunique(dropna=True)
    inconsistent = labels_per_group[labels_per_group > 1]
    return {
        "n_groups": int(len(labels_per_group)),
        "groups_with_conflicting_labels": int(len(inconsistent)),
        "examples": list(inconsistent.index[:5]),
    }


def check_bilingual_pairs(frame: pd.DataFrame) -> dict[str, Any]:
    """Every pair_id should contain exactly one row per language, in one group."""
    paired = frame[frame["pair_id"].notna()]
    if paired.empty:
        return {"n_pairs": 0}
    per_pair = paired.groupby("pair_id").agg(
        n_rows=("sample_id", "size"),
        n_languages=("language", "nunique"),
        n_groups=("group_id", "nunique"),
        n_labels=("label_binary", "nunique"),
    )
    return {
        "n_pairs": int(len(per_pair)),
        "pairs_not_exactly_two_rows": int((per_pair["n_rows"] != 2).sum()),
        "pairs_not_two_languages": int((per_pair["n_languages"] != 2).sum()),
        "pairs_spanning_multiple_groups": int((per_pair["n_groups"] > 1).sum()),
        "pairs_with_conflicting_labels": int((per_pair["n_labels"] > 1).sum()),
    }


def check_templates(frame: pd.DataFrame) -> dict[str, Any]:
    """Groups large enough to indicate templating, which forces group-level splits."""
    sizes = frame.groupby("group_id").size().sort_values(ascending=False)
    large = sizes[sizes > 1]
    return {
        "n_groups": int(len(sizes)),
        "n_groups_with_multiple_rows": int(len(large)),
        "largest_groups": [
            {
                "group_id": gid,
                "n_rows": int(n),
                "source": frame.loc[frame["group_id"] == gid, "source"].iloc[0],
            }
            for gid, n in sizes.head(5).items()
        ],
        "max_group_size": int(sizes.max()),
        "pct_rows_in_multi_row_groups": round(100.0 * int(large.sum()) / max(len(frame), 1), 2),
    }


def distributions(frame: pd.DataFrame) -> dict[str, Any]:
    """Source, language and provenance distributions."""
    return {
        "by_source": frame["source"].value_counts().to_dict(),
        "by_language": frame["language"].value_counts().to_dict(),
        "by_synthetic": {str(k): int(v) for k, v in frame["is_synthetic"].value_counts(dropna=False).items()},
        "by_source_language": {
            f"{source}/{language}": int(len(group))
            for (source, language), group in frame.groupby(["source", "language"])
        },
    }


def proposed_removals(frame: pd.DataFrame, checks: dict[str, Any]) -> list[dict[str, Any]]:
    """What a cleaning pass would remove, and why. NOT applied here."""
    removals = []
    empty = int((frame["text"].astype(str).str.strip() == "").sum())
    if empty:
        removals.append({"reason": "empty text", "n_rows": empty, "applied": False})
    unlabelled = int(frame["label_binary"].isna().sum())
    if unlabelled:
        removals.append(
            {
                "reason": "label_binary is null (source vocabulary not yet mapped)",
                "n_rows": unlabelled,
                "applied": False,
            }
        )
    if checks["duplicates"]["cross_source_duplicate_texts"]:
        removals.append(
            {
                "reason": "identical text appears in more than one source",
                "n_texts": checks["duplicates"]["cross_source_duplicate_texts"],
                "applied": False,
                "note": "grouping by normalized text already prevents these from splitting apart",
            }
        )
    return removals


def run_quality_checks() -> dict[str, Any]:
    """Run every check and write the report."""
    frame = pd.read_parquet(CANONICAL_PATH)
    checks: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "empty_and_missing": check_empty_and_missing(frame),
        "duplicates": check_duplicates(frame),
        "label_consistency": check_label_consistency(frame),
        "bilingual_pairs": check_bilingual_pairs(frame),
        "templates": check_templates(frame),
        "distributions": distributions(frame),
    }
    checks["proposed_removals"] = proposed_removals(frame, checks)
    checks["note"] = "No rows were removed. Removals are proposals requiring a ruling."

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, default=str)
    logger.info("wrote %s", REPORT_PATH)
    return checks


if __name__ == "__main__":
    result = run_quality_checks()
    print(json.dumps({k: v for k, v in result.items() if k != "distributions"}, indent=2, default=str)[:3000])
