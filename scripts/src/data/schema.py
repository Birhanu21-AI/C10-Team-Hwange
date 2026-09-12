"""The canonical schema shared by every source (Phase 4).

Two rules govern it:
  * `label_original` always survives, so no source annotation is lost to the
    binary mapping.
  * `group_id` is the unit that splitting must respect. Rows sharing a group_id
    (a template, a translation pair) can never land on opposite sides of a split.
"""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS: list[str] = [
    "sample_id",          # globally unique, prefixed by source
    "text",               # the model input, one language per row
    "text_original",      # text before any normalization
    "language",           # ISO-ish code: en, amh
    "source",             # hatexplain | ubuntu | toxigen | afrihate
    "source_file",        # which file/config/split within the source
    "label_binary",       # 0 = safe, 1 = harmful; nullable when undecidable
    "label_original",     # the source's own label, verbatim
    "label_type",         # how label_original was produced
    "toxicity_type",      # Data Card 5: hate_speech | harassment | abuse |
                          # offensive_language | none
    "is_ethnic_target",   # Data Card 6: ethnic-hate sub-focus slice
    "harm_type",          # Ubuntu taxonomy, else null
    "target_type",        # individual | group | none
    "target_group",       # targeted group, source vocabulary
    "severity",           # verbatim; see severity_scheme
    "severity_scheme",    # which vocabulary `severity` belongs to
    "context",            # situational context, Ubuntu synthetic only
    "rationale",          # free-text justification where the source provides one
    "is_synthetic",       # per row, never per dataset
    "annotation_quality", # unanimous | majority | tie | human_rated | template_generated
    "pair_id",            # links translations of the same item
    "group_id",           # LEAKAGE UNIT — splits must not break this
    "official_split",     # the source's own split, where one exists
]

BINARY_LABELS = {0: "safe", 1: "harmful"}


def empty_frame() -> pd.DataFrame:
    """An empty frame with the canonical columns, for loaders to fill."""
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Add any missing canonical columns as null and order them consistently."""
    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[CANONICAL_COLUMNS]


def validate(frame: pd.DataFrame) -> list[str]:
    """Return schema violations. Empty list means the frame is well formed."""
    problems: list[str] = []
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
        return problems

    if frame["sample_id"].duplicated().any():
        n = int(frame["sample_id"].duplicated().sum())
        problems.append(f"{n} duplicate sample_id values")
    if frame["group_id"].isna().any():
        problems.append(f"{int(frame['group_id'].isna().sum())} rows with null group_id")
    if frame["text"].isna().any() or (frame["text"].astype(str).str.strip() == "").any():
        problems.append("rows with empty text")
    bad_labels = set(frame["label_binary"].dropna().unique()) - {0, 1}
    if bad_labels:
        problems.append(f"label_binary contains non-binary values: {bad_labels}")
    if frame["label_original"].isna().all():
        problems.append("label_original is entirely null — source annotation was lost")
    return problems
