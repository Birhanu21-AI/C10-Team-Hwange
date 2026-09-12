"""Phase 3: dataset-specific loaders.

Each loader returns the canonical schema while preserving its source's own
annotation in `label_original`. Label mappings that Phase 2 flagged as unresolved
are read from configs/data.yaml, never hard-coded here.

`group_id` is derived from normalized text (for Ubuntu, from the English pivot
sentence) so that duplicates, templates and translation pairs form one leakage
unit that Phase 6 must not split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from typing import Any, Optional

import pandas as pd

from src import paths
from src.data import schema
from src.data.profiling import normalize_text

logger = logging.getLogger(__name__)


def group_id_for(text: str) -> str:
    """Stable leakage-group id derived from normalized text."""
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
    return f"g_{digest[:12]}"


# --------------------------------------------------------------------------- #
# HateXplain
# --------------------------------------------------------------------------- #

def load_hatexplain(config: dict[str, Any]) -> pd.DataFrame:
    """Load HateXplain, resolving annotators by the configured policy."""
    policy = config["hatexplain"]
    if not paths.HATEXPLAIN_DATA.exists():
        logger.warning("HateXplain not found at %s — skipping", paths.HATEXPLAIN_DATA)
        return schema.empty_frame()

    with paths.HATEXPLAIN_DATA.open(encoding="utf-8") as handle:
        records = json.load(handle)

    splits: dict[str, str] = {}
    if paths.HATEXPLAIN_SPLITS.exists() and policy.get("use_official_splits", True):
        with paths.HATEXPLAIN_SPLITS.open(encoding="utf-8") as handle:
            for split_name, post_ids in json.load(handle).items():
                splits.update({post_id: split_name for post_id in post_ids})

    offensive_policy = policy["offensive_maps_to"]
    tie_policy = policy["tie_posts"]
    toxicity_type_map = policy.get("toxicity_type_map", {})
    ethnic_targets = {t.casefold() for t in config.get("ethnic_targets", [])}

    rows: list[dict[str, Any]] = []
    for post_id, record in records.items():
        labels = [a["label"] for a in record["annotators"]]
        counts = Counter(labels).most_common()
        is_tie = len(counts) > 1 and counts[0][1] == counts[1][1]

        if is_tie:
            if tie_policy == "exclude":
                continue
            majority_label = "no_majority"
            label_binary: Optional[int] = 1 if tie_policy == "include_as_harmful" else 0
            quality = "tie"
        else:
            majority_label = counts[0][0]
            quality = "unanimous" if len(counts) == 1 else "majority"
            if majority_label == "hatespeech":
                label_binary = 1
            elif majority_label == "normal":
                label_binary = 0
            else:  # offensive
                if offensive_policy == "exclude":
                    continue
                label_binary = 1 if offensive_policy == "harmful" else 0

        targets = Counter(t for a in record["annotators"] for t in (a.get("target") or []))
        named_targets = [t for t, _ in targets.most_common(3) if t.casefold() != "none"]
        text = " ".join(record["post_tokens"])
        rows.append(
            {
                "sample_id": f"hatexplain_{post_id}",
                "text": text,
                "text_original": text,
                "language": "en",
                "source": "hatexplain",
                "source_file": "Data/dataset.json",
                "label_binary": label_binary,
                "label_original": majority_label,
                "label_type": "majority_vote_of_3_human_annotators",
                "target_group": ", ".join(named_targets) or None,
                "toxicity_type": toxicity_type_map.get(majority_label, "none"),
                "is_ethnic_target": any(t.casefold() in ethnic_targets for t in named_targets),
                "is_synthetic": False,
                "annotation_quality": quality,
                "group_id": group_id_for(text),
                "official_split": splits.get(post_id),
            }
        )

    frame = schema.conform(pd.DataFrame(rows))
    logger.info(
        "hatexplain: %d rows (offensive=%s, ties=%s)", len(frame), offensive_policy, tie_policy
    )
    return frame


# --------------------------------------------------------------------------- #
# Ubuntu
# --------------------------------------------------------------------------- #

UBUNTU_LABEL_MAP = {"safe": 0, "hate_speech": 1}


def load_ubuntu(config: dict[str, Any]) -> pd.DataFrame:
    """Load both Ubuntu files as one row per language, sharing pair_id and group_id.

    English and Amharic are never concatenated into one input. The group_id of an
    Amharic row is derived from its English pivot sentence, so a template's two
    languages and both files form a single indivisible leakage unit.
    """
    files = {
        "ubuntu_main": (paths.UBUNTU_MAIN_CSV, "zero_low_high"),
        "ubuntu_synthetic": (paths.UBUNTU_SYNTHETIC_CSV, "low_medium_high"),
    }

    rows: list[dict[str, Any]] = []
    for source_file, (path, severity_scheme) in files.items():
        if not path.exists():
            logger.warning("Ubuntu file missing: %s", path)
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        bilingual = "label_en" in frame.columns

        for _, record in frame.iterrows():
            pair_id = f"{source_file}_{record['id']}"
            english = str(record["text_en"])
            group_id = group_id_for(english)  # English pivot for BOTH languages
            label_en = str(record["label_en"] if bilingual else record["label"])
            harm_type = str(record["ubuntu_harm_type_en"] if bilingual else record["ubuntu_harm_type"])
            target_type = str(record["relation_target_en"] if bilingual else record["relation_target"])
            severity = str(record["severity_en"] if bilingual else record["severity"])
            binary = UBUNTU_LABEL_MAP.get(label_en)

            shared = {
                "source": "ubuntu",
                "source_file": source_file,
                "label_binary": binary,
                "label_type": "template_metadata_assigned_at_generation",
                "harm_type": harm_type,
                # Ubuntu's relational taxonomy is not the Data Card's toxicity_type
                # vocabulary; the harmful classes are recorded as hate_speech only at
                # the binary level, with the relational category kept in harm_type.
                "toxicity_type": "hate_speech" if binary == 1 else "none",
                "is_ethnic_target": False,
                "target_type": target_type,
                "severity": severity,
                "severity_scheme": severity_scheme,
                "context": str(record["context"]) if "context" in frame.columns else None,
                "is_synthetic": True,
                "annotation_quality": "template_generated",
                "pair_id": pair_id,
                "group_id": group_id,
                "official_split": None,
            }
            rows.append(
                {
                    **shared,
                    "sample_id": f"ubuntu_{source_file}_{record['id']}_en",
                    "text": english,
                    "text_original": english,
                    "language": "en",
                    "label_original": label_en,
                    "rationale": record.get("rationale_en"),
                }
            )
            amharic = str(record["text_am"])
            rows.append(
                {
                    **shared,
                    "sample_id": f"ubuntu_{source_file}_{record['id']}_am",
                    "text": amharic,
                    "text_original": amharic,
                    "language": "amh",
                    # the Amharic label string where the file provides one
                    "label_original": str(record["label_am"]) if bilingual else label_en,
                    "rationale": record.get("rationale_am"),
                }
            )

    frame = schema.conform(pd.DataFrame(rows))
    logger.info("ubuntu: %d rows (role=%s)", len(frame), config["ubuntu"]["role"])
    return frame


# --------------------------------------------------------------------------- #
# ToxiGen
# --------------------------------------------------------------------------- #

def load_toxigen(config: dict[str, Any]) -> pd.DataFrame:
    """Load the human-annotated ToxiGen config and binarize by the configured threshold."""
    policy = config["toxigen"]
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets not installed — skipping ToxiGen")
        return schema.empty_frame()

    field = policy["binarize_field"]
    threshold = float(policy["threshold"])
    toxicity_type = policy.get("toxicity_type", "hate_speech")
    ethnic_targets = {t.casefold() for t in config.get("ethnic_targets", [])}
    # ToxiGen target_group strings are phrases ("jewish folks"), so match on substring.
    def _is_ethnic(group: Any) -> bool:
        text_value = str(group or "").casefold()
        return any(term in text_value for term in ethnic_targets)
    rows: list[dict[str, Any]] = []

    for config_name in policy["configs"]:
        try:
            dataset = load_dataset(paths.TOXIGEN_HUB_ID, config_name)
        except Exception as exc:
            logger.warning("ToxiGen config %s unavailable: %s", config_name, exc)
            continue

        for split_name, split in dataset.items():
            frame = split.to_pandas()
            for index, record in frame.iterrows():
                text = str(record["text"])
                rating = record.get(field)
                method = str(record.get("actual_method", ""))
                rows.append(
                    {
                        "sample_id": f"toxigen_{config_name}_{split_name}_{index}",
                        "text": text,
                        "text_original": text,
                        "language": "en",
                        "source": "toxigen",
                        "source_file": f"{config_name}/{split_name}",
                        "label_binary": None if pd.isna(rating) else int(float(rating) > threshold),
                        "label_original": None if pd.isna(rating) else f"{field}={rating}",
                        "label_type": f"continuous_1_5_mean_of_3_human_raters_binarized_at_{threshold}",
                        "target_group": record.get("target_group"),
                        # a harmful row is hate speech; a benign row is not toxic at all
                        "toxicity_type": (
                            toxicity_type
                            if (not pd.isna(rating) and float(rating) > threshold)
                            else "none"
                        ),
                        "is_ethnic_target": _is_ethnic(record.get("target_group")),
                        # provenance is per row: some rows are human-written
                        "is_synthetic": method != "human",
                        "annotation_quality": "human_rated",
                        "group_id": group_id_for(text),
                        "official_split": split_name,
                    }
                )

    frame = schema.conform(pd.DataFrame(rows))
    logger.info("toxigen: %d rows (threshold=%s)", len(frame), threshold)
    return frame


# --------------------------------------------------------------------------- #
# AfriHate
# --------------------------------------------------------------------------- #

def load_afrihate(config: dict[str, Any]) -> pd.DataFrame:
    """Load AfriHate if access has been granted; otherwise return empty with a warning.

    The label vocabulary is NOT mapped here. AfriHate distinguishes hate from
    abuse, and that mapping is an open decision recorded in the Phase 2 report.
    """
    policy = config["afrihate"]
    if not policy.get("enabled"):
        logger.warning("AfriHate disabled in config (gated dataset — needs HF_TOKEN)")
        return schema.empty_frame()
    # get_token() checks HF_TOKEN, HUGGING_FACE_HUB_TOKEN and the machine-wide
    # cache written by `hf auth login`, so no token needs to live in this repo.
    try:
        from huggingface_hub import get_token

        has_token = bool(get_token())
    except ImportError:
        has_token = bool(os.environ.get("HF_TOKEN"))
    if not has_token:
        logger.warning("AfriHate enabled but no HuggingFace token found — run `hf auth login`")
        return schema.empty_frame()

    try:
        from datasets import load_dataset
    except ImportError:
        return schema.empty_frame()

    label_map = policy["label_map"]
    toxicity_type_map = policy.get("toxicity_type_map", {})

    rows: list[dict[str, Any]] = []
    for language in policy["languages"]:
        try:
            dataset = load_dataset(paths.AFRIHATE_HUB_ID, language)
        except Exception as exc:
            logger.warning("AfriHate %s unavailable: %s", language, exc)
            continue
        for split_name, split in dataset.items():
            frame = split.to_pandas()
            text_col = next((c for c in ("tweet", "text") if c in frame.columns), None)
            label_col = next((c for c in ("label", "labels", "class") if c in frame.columns), None)
            if text_col is None or label_col is None:
                logger.warning("AfriHate %s: unexpected columns %s", language, list(frame.columns))
                continue
            unknown = set(frame[label_col].unique()) - set(label_map)
            if unknown:
                # never guess at an unmapped class: leave it unlabelled and say so
                logger.warning("AfriHate %s: unmapped labels left as null: %s", language, unknown)
            for index, record in frame.iterrows():
                text = str(record[text_col])
                original = str(record[label_col])
                rows.append(
                    {
                        "sample_id": f"afrihate_{language}_{split_name}_{index}",
                        "text": text,
                        "text_original": text,
                        "language": language,
                        "source": "afrihate",
                        "source_file": f"{language}/{split_name}",
                        "label_binary": label_map.get(original),
                        "label_original": original,
                        "label_type": "human_annotated_native_speakers_3_way",
                        "toxicity_type": toxicity_type_map.get(original),
                        # AfriHate provides no target-group column
                        "is_ethnic_target": None,
                        "is_synthetic": False,
                        "annotation_quality": "human_annotated",
                        "group_id": group_id_for(text),
                        "official_split": "val" if split_name == "validation" else split_name,
                    }
                )

    frame = schema.conform(pd.DataFrame(rows))
    logger.info("afrihate: %d rows", len(frame))
    return frame


LOADERS = {
    "hatexplain": load_hatexplain,
    "ubuntu": load_ubuntu,
    "toxigen": load_toxigen,
    "afrihate": load_afrihate,
}
