"""Generic dataset profiling helpers used by the Phase 2 inspection.

These functions describe data; they never modify, filter or relabel it. Any
judgement about what a label *means* is left to the research decision-maker.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

# Values treated as categorical enough to enumerate in full.
MAX_ENUMERATED_VALUES = 25

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalization used for near-duplicate detection only.

    NFKC + casefold + whitespace collapse. Deliberately conservative: it does not
    strip punctuation or diacritics, which carry meaning in Amharic.
    """
    normalized = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    return _WHITESPACE.sub(" ", normalized)


def value_distribution(values: Iterable[Any], limit: int = MAX_ENUMERATED_VALUES) -> dict[str, Any]:
    """Frequency of each value, with the full set enumerated when small enough."""
    counts = Counter("" if pd.isna(v) else str(v) for v in values)
    ordered = counts.most_common()
    return {
        "n_distinct": len(counts),
        "fully_enumerated": len(counts) <= limit,
        "counts": dict(ordered[:limit]),
    }


def profile_column(series: pd.Series, limit: int = MAX_ENUMERATED_VALUES) -> dict[str, Any]:
    """Describe one column: dtype, missingness, cardinality, distribution."""
    n_missing = int(series.isna().sum())
    empty_strings = int((series.astype(str).str.strip() == "").sum())
    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "n_missing": n_missing,
        "pct_missing": round(100.0 * n_missing / max(len(series), 1), 3),
        "n_empty_string": empty_strings,
        "n_distinct": int(series.nunique(dropna=True)),
    }
    profile.update(value_distribution(series.dropna(), limit=limit))
    if profile["n_distinct"] > limit:
        profile["examples"] = [str(v) for v in series.dropna().unique()[:3]]
    return profile


def duplicate_analysis(series: pd.Series) -> dict[str, Any]:
    """Exact and normalized duplication within a text column.

    `repetition_factor` is rows per distinct value: 1.0 means every row is unique,
    121.2 means each distinct string appears ~121 times. High values indicate
    template generation and make row-level splits leaky.
    """
    texts = series.dropna().astype(str)
    n_rows = len(texts)
    n_exact = int(texts.nunique())
    n_normalized = int(texts.map(normalize_text).nunique())
    most_common = Counter(texts).most_common(5)
    return {
        "n_rows": n_rows,
        "n_distinct_exact": n_exact,
        "n_distinct_normalized": n_normalized,
        "n_exact_duplicate_rows": n_rows - n_exact,
        "pct_duplicate_rows": round(100.0 * (n_rows - n_exact) / max(n_rows, 1), 2),
        "repetition_factor": round(n_rows / max(n_exact, 1), 2),
        "most_repeated": [{"text": t[:120], "count": c} for t, c in most_common],
    }


def overlap(left: Iterable[str], right: Iterable[str], normalize: bool = True) -> dict[str, Any]:
    """Set overlap between two text collections, for cross-dataset leakage checks."""
    transform = normalize_text if normalize else str
    left_set = {transform(t) for t in left}
    right_set = {transform(t) for t in right}
    shared = left_set & right_set
    return {
        "n_left_distinct": len(left_set),
        "n_right_distinct": len(right_set),
        "n_shared": len(shared),
        "pct_of_left_shared": round(100.0 * len(shared) / max(len(left_set), 1), 2),
        "examples": sorted(shared)[:5],
    }


def example_records(
    frame: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """A small, reproducibly sampled set of records for eyeballing."""
    subset = frame if columns is None else frame[list(columns)]
    sample = subset.sample(n=min(n, len(subset)), random_state=seed)
    return [
        {k: (str(v)[:200] if v is not None else None) for k, v in row.items()}
        for row in sample.to_dict(orient="records")
    ]


def profile_frame(
    frame: pd.DataFrame,
    text_columns: Sequence[str] = (),
    limit: int = MAX_ENUMERATED_VALUES,
) -> dict[str, Any]:
    """Full profile of a dataframe: shape, per-column stats, duplication of text."""
    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": list(frame.columns),
        "column_profiles": {col: profile_column(frame[col], limit=limit) for col in frame.columns},
        "duplication": {col: duplicate_analysis(frame[col]) for col in text_columns if col in frame},
        "examples": example_records(frame, n=3),
    }
