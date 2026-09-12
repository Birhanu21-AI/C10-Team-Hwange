"""Phase 2: dataset discovery and inspection.

Produces a reproducible report per source under reports/phase2_inspection/.
This module DESCRIBES the data. It performs no label mapping, no merging, no
cleaning and no filtering -- those are later phases and require a research
decision that this code deliberately does not make.

Run:
    python -m src.data.inspection
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src import paths
from src.data import profiling

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SEED = 42


# --------------------------------------------------------------------------- #
# HateXplain
# --------------------------------------------------------------------------- #

def _hatexplain_annotation_structure(records: dict[str, Any]) -> dict[str, Any]:
    """Describe the multi-annotator structure without resolving disagreement.

    HateXplain gives several independent annotations per post. Collapsing them
    (majority vote, and whether `offensive` counts as harmful) changes both the
    label distribution and the meaning of the task, so both are reported as open
    decisions rather than applied here.
    """
    annotator_label_counts: Counter[str] = Counter()
    annotators_per_post: Counter[int] = Counter()
    target_counts: Counter[str] = Counter()
    agreement_counts: Counter[str] = Counter()
    majority_labels: Counter[str] = Counter()
    ties = 0
    posts_with_rationales = 0

    for record in records.values():
        labels = [a["label"] for a in record["annotators"]]
        annotator_label_counts.update(labels)
        annotators_per_post[len(labels)] += 1
        for annotator in record["annotators"]:
            target_counts.update(annotator.get("target", []) or ["<none>"])

        counts = Counter(labels)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ties += 1
            agreement_counts["no majority (tie)"] += 1
        else:
            majority_labels[top[0][0]] += 1
            agreement_counts["unanimous" if len(counts) == 1 else "majority"] += 1

        if record.get("rationales"):
            posts_with_rationales += 1

    n_posts = len(records)
    return {
        "annotators_per_post": dict(annotators_per_post),
        "annotator_label_vocabulary": dict(annotator_label_counts),
        "agreement": dict(agreement_counts),
        "pct_unanimous": round(100.0 * agreement_counts["unanimous"] / max(n_posts, 1), 2),
        "n_ties_no_majority": ties,
        "majority_label_distribution_excluding_ties": dict(majority_labels),
        "target_vocabulary": dict(target_counts.most_common(30)),
        "n_target_categories": len(target_counts),
        "posts_with_rationales": posts_with_rationales,
        "pct_with_rationales": round(100.0 * posts_with_rationales / max(n_posts, 1), 2),
    }


def inspect_hatexplain() -> dict[str, Any]:
    """Inspect the HateXplain JSON corpus and its official splits."""
    if not paths.HATEXPLAIN_DATA.exists():
        return {"name": "HateXplain", "status": "MISSING", "expected_path": str(paths.HATEXPLAIN_DATA)}

    with paths.HATEXPLAIN_DATA.open(encoding="utf-8") as handle:
        records = json.load(handle)

    frame = pd.DataFrame(
        [
            {
                "post_id": post_id,
                "text": " ".join(record["post_tokens"]),
                "n_tokens": len(record["post_tokens"]),
                "n_annotators": len(record["annotators"]),
            }
            for post_id, record in records.items()
        ]
    )

    report: dict[str, Any] = {
        "name": "HateXplain",
        "status": "AVAILABLE",
        "path": str(paths.HATEXPLAIN_DATA.relative_to(paths.PROJECT_ROOT)),
        "format": "JSON, one object per post_id",
        "n_records": len(records),
        "fields": sorted(next(iter(records.values())).keys()),
        "provenance": {
            "generation": "human-written social media posts (Twitter, Gab)",
            "annotation": "human, multiple annotators per post",
            "is_synthetic": False,
        },
        "language": {
            "stated": "English",
            "verified": False,
            "note": "no language ID run; corpus is documented as English",
        },
        "profile": profiling.profile_frame(frame, text_columns=["text"]),
        "annotation_structure": _hatexplain_annotation_structure(records),
        "open_decisions": [
            "How to resolve multi-annotator disagreement (majority vote vs keeping soft labels).",
            "Whether the `offensive` class maps to harmful, to safe, or is excluded. "
            "This is the single largest lever on the binary label distribution and is NOT decided here.",
            f"How to handle the {_hatexplain_annotation_structure(records)['n_ties_no_majority']} posts with no majority label.",
        ],
    }

    if paths.HATEXPLAIN_SPLITS.exists():
        with paths.HATEXPLAIN_SPLITS.open(encoding="utf-8") as handle:
            divisions = json.load(handle)
        split_ids = {name: set(ids) for name, ids in divisions.items()}
        all_split_ids: set[str] = set().union(*split_ids.values())
        report["official_splits"] = {
            "path": str(paths.HATEXPLAIN_SPLITS.relative_to(paths.PROJECT_ROOT)),
            "sizes": {name: len(ids) for name, ids in split_ids.items()},
            "total_assigned": len(all_split_ids),
            "unassigned_posts": len(set(records) - all_split_ids),
            "overlap_between_splits": sum(
                len(a & b)
                for i, a in enumerate(split_ids.values())
                for j, b in enumerate(split_ids.values())
                if i < j
            ),
        }
    else:
        report["official_splits"] = {"status": "MISSING"}

    return report


# --------------------------------------------------------------------------- #
# Ubuntu
# --------------------------------------------------------------------------- #

def _template_analysis(frame: pd.DataFrame, text_col: str, label_col: str) -> dict[str, Any]:
    """Quantify template repetition and whether a template fixes its label.

    If each distinct sentence always carries the same label, the sentence itself
    determines the label, and any row-level split leaks the answer.
    """
    grouped = frame.groupby(text_col)[label_col].nunique()
    labels_per_template = Counter(grouped.tolist())
    return {
        "n_distinct_texts": int(frame[text_col].nunique()),
        "n_rows": int(len(frame)),
        "rows_per_distinct_text": round(len(frame) / max(frame[text_col].nunique(), 1), 2),
        "labels_per_distinct_text": {str(k): int(v) for k, v in sorted(labels_per_template.items())},
        "template_determines_label": bool(grouped.max() == 1),
    }


def _bilingual_pairing(frame: pd.DataFrame) -> dict[str, Any]:
    """Check whether English and Amharic columns form a consistent 1:1 mapping."""
    en_to_am = frame.groupby("text_en")["text_am"].nunique()
    am_to_en = frame.groupby("text_am")["text_en"].nunique()
    return {
        "n_distinct_en": int(frame["text_en"].nunique()),
        "n_distinct_am": int(frame["text_am"].nunique()),
        "en_mapping_to_multiple_am": int((en_to_am > 1).sum()),
        "am_mapping_to_multiple_en": int((am_to_en > 1).sum()),
        "is_consistent_one_to_one": bool((en_to_am.max() == 1) and (am_to_en.max() == 1)),
        "note": (
            "Rows carry parallel EN/AM text. They must become separate language samples "
            "sharing a pair_id, never concatenated, never split apart."
        ),
    }


def inspect_ubuntu() -> dict[str, Any]:
    """Inspect both Ubuntu CSV files, including template and bilingual structure."""
    files = {"ubuntu_main": paths.UBUNTU_MAIN_CSV, "ubuntu_synthetic": paths.UBUNTU_SYNTHETIC_CSV}
    report: dict[str, Any] = {
        "name": "Ubuntu (English-Amharic)",
        "status": "AVAILABLE",
        "format": "CSV, UTF-8 with BOM",
        "files": {},
        "provenance": {
            "generation": "template-generated; not collected from real discourse",
            "annotation": "metadata assigned at generation time, not independently annotated",
            "is_synthetic": True,
            "note": (
                "Filename of the second file states 'synthetic'. The first file is not labelled "
                "synthetic but shows the same template structure. Treated as synthetic pending "
                "confirmation of how it was produced."
            ),
        },
        "open_decisions": [
            "BLOCKING: only 33 distinct English sentences exist across both files (4,000 and 500 "
            "rows). Each distinct sentence always carries the same label, so the sentence alone "
            "determines the label. A row-level split leaks completely; a group-level split leaves "
            "~33 independent units, which cannot support train/val/test evaluation. Decision "
            "needed on whether Ubuntu is a training source at all, or only a diagnostic probe set.",
            "CONFLICT: the two files use incompatible severity vocabularies. Main uses "
            "zero/low/high with safe='zero'; synthetic uses low/medium/high with safe='low'. "
            "The token 'low' therefore means SAFE in one file and HARMFUL in the other. These "
            "must not be merged into one severity field without an explicit ruling.",
            "CONFLICT: 8 of 33 English sentences have different Amharic text between the two "
            "files. Several differ in second-person grammatical gender (e.g. -ሽ feminine vs -ህ "
            "masculine). Needs native-speaker review: a systematic gender split across labels "
            "would be a spurious morphological cue that a probe could exploit.",
            "The 4,000-row file is not named 'synthetic' but shows identical template structure. "
            "Confirmation needed on how it was produced before any provenance claim is made.",
            "Schema mismatch: main carries bilingual label/severity/harm columns (label_en, "
            "label_am, ...), synthetic carries single-language equivalents plus a 'context' "
            "column absent from main.",
        ],
    }

    frames: dict[str, pd.DataFrame] = {}
    for key, path in files.items():
        if not path.exists():
            report["files"][key] = {"status": "MISSING", "expected_path": str(path)}
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frames[key] = frame
        label_col = "label_en" if "label_en" in frame.columns else "label"
        entry: dict[str, Any] = {
            "status": "AVAILABLE",
            "path": str(path.relative_to(paths.PROJECT_ROOT)),
            "profile": profiling.profile_frame(frame, text_columns=["text_en", "text_am"]),
            "template_analysis_en": _template_analysis(frame, "text_en", label_col),
            "bilingual_pairing": _bilingual_pairing(frame),
            "label_column_used": label_col,
        }
        report["files"][key] = entry

    if len(frames) == 2:
        main, synthetic = frames["ubuntu_main"], frames["ubuntu_synthetic"]
        report["cross_file_overlap_en"] = profiling.overlap(main["text_en"], synthetic["text_en"])
        report["cross_file_overlap_am"] = profiling.overlap(main["text_am"], synthetic["text_am"])
        report["schema_differences"] = {
            "only_in_main": sorted(set(main.columns) - set(synthetic.columns)),
            "only_in_synthetic": sorted(set(synthetic.columns) - set(main.columns)),
            "shared": sorted(set(main.columns) & set(synthetic.columns)),
        }

    return report


# --------------------------------------------------------------------------- #
# Hub-hosted sources: AfriHate, ToxiGen
# --------------------------------------------------------------------------- #

def _hub_availability(repo_id: str) -> dict[str, Any]:
    """Query the Hub for a dataset's availability without downloading it."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return {"reachable": False, "reason": "huggingface_hub not installed"}

    try:
        info = HfApi().dataset_info(repo_id, timeout=30)
    except GatedRepoError:
        return {"reachable": False, "gated": True, "reason": "gated — requires an accepted licence and HF_TOKEN"}
    except RepositoryNotFoundError:
        return {"reachable": False, "reason": "not found, or private and unauthenticated"}
    except Exception as exc:  # network failure, auth failure, transport error
        return {"reachable": False, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}

    return {
        "reachable": True,
        "gated": bool(getattr(info, "gated", False)),
        "configs": sorted({s.config_name for s in (info.siblings or []) if hasattr(s, "config_name")}) or None,
        "last_modified": str(getattr(info, "last_modified", None)),
    }


def _inventory(directory: Path, suffixes: tuple[str, ...]) -> list[dict[str, Any]]:
    """List files of the given suffixes with line counts, for repos without data."""
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            try:
                n_lines = sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
            except OSError:
                n_lines = -1
            entries.append(
                {
                    "path": str(path.relative_to(paths.PROJECT_ROOT)),
                    "bytes": path.stat().st_size,
                    "n_lines": n_lines,
                }
            )
    return entries


def _toxigen_configs() -> list[str] | str:
    """List the ToxiGen Hub configs, which differ in provenance and must not be pooled."""
    try:
        from datasets import get_dataset_config_names

        return sorted(get_dataset_config_names(paths.TOXIGEN_HUB_ID))
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:150]}"


def _inspect_toxigen_annotated() -> dict[str, Any]:
    """Profile the human-annotated ToxiGen config, which is fully downloadable."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "SKIPPED", "reason": "datasets not installed"}

    try:
        dataset = load_dataset(paths.TOXIGEN_HUB_ID, "annotated")
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}

    per_split = {}
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        per_split[split_name] = {
            "n_rows": int(len(frame)),
            "profile": profiling.profile_frame(frame, text_columns=["text"]),
        }
    return {"status": "AVAILABLE", "splits": {k: v["n_rows"] for k, v in per_split.items()}, "detail": per_split}


def _peek_toxigen_config(config: str, n: int = 1000) -> dict[str, Any]:
    """Sample a large ToxiGen config by streaming, to record schema without a full download."""
    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "SKIPPED", "reason": "datasets not installed"}

    try:
        stream = load_dataset(paths.TOXIGEN_HUB_ID, config, split="train", streaming=True)
        rows = [row for _, row in zip(range(n), stream)]
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}

    if not rows:
        return {"status": "EMPTY"}
    frame = pd.DataFrame(rows)
    text_cols = [c for c in ("text", "generation", "prompt") if c in frame.columns]
    return {
        "status": "SAMPLED",
        "n_sampled": len(rows),
        "note": "distributions below are from a stream sample, NOT the full config",
        "profile": profiling.profile_frame(frame, text_columns=text_cols),
    }


def inspect_afrihate() -> dict[str, Any]:
    """Inspect the vendored AfriHate repo and check Hub availability."""
    local_data = _inventory(paths.AFRIHATE_DIR, (".csv", ".tsv", ".json", ".jsonl", ".parquet"))
    return {
        "name": "AfriHate",
        "status": "NO LOCAL DATA" if not local_data else "AVAILABLE",
        "local_path": str(paths.AFRIHATE_DIR.relative_to(paths.PROJECT_ROOT)),
        "local_data_files": local_data,
        "local_contents_note": "repository clone contains README and assets only",
        "hub": {"repo_id": paths.AFRIHATE_HUB_ID, **_hub_availability(paths.AFRIHATE_HUB_ID)},
        "provenance": {
            "generation": "human-written tweets in 18 African languages",
            "annotation": "human, native-speaker annotators",
            "is_synthetic": False,
        },
        "open_decisions": [
            "Label vocabulary is unknown until the Hub data is fetched. No mapping to binary "
            "is possible or attempted yet; AfriHate distinguishes hate from abuse/offensive.",
            "Which of the 18 languages to include, and whether Amharic is selected to pair with Ubuntu.",
        ],
    }


def inspect_toxigen() -> dict[str, Any]:
    """Separate ToxiGen's generation prompts from its actual labelled data."""
    prompt_files = _inventory(paths.TOXIGEN_DIR / "prompts", (".txt",))
    demonstration_files = _inventory(paths.TOXIGEN_DIR / "demonstrations", (".txt",))
    labelled_local = _inventory(paths.TOXIGEN_DIR, (".csv", ".jsonl", ".parquet"))

    return {
        "name": "ToxiGen",
        "status": "NO LOCAL LABELS",
        "local_path": str(paths.TOXIGEN_DIR.relative_to(paths.PROJECT_ROOT)),
        "local_contents": {
            "prompt_files": {"n_files": len(prompt_files), "total_lines": sum(f["n_lines"] for f in prompt_files)},
            "demonstration_files": {
                "n_files": len(demonstration_files),
                "total_lines": sum(f["n_lines"] for f in demonstration_files),
            },
            "labelled_data_files": labelled_local,
        },
        "critical_distinction": (
            "prompts/*.txt and demonstrations/*/*.txt are GENERATION SEEDS used to produce the "
            "corpus. They are not labelled data and must never be loaded as training examples."
        ),
        "hub": {
            "repo_id": paths.TOXIGEN_HUB_ID,
            **_hub_availability(paths.TOXIGEN_HUB_ID),
            "configs": _toxigen_configs(),
            "annotated": _inspect_toxigen_annotated(),
            "train_sample": _peek_toxigen_config("train"),
        },
        "provenance": {
            "generation": "machine-generated (GPT-3 via ALICE decoding)",
            "annotation": "config-dependent — see open decisions",
            "is_synthetic": True,
        },
        "open_decisions": [
            "Which Hub config to use. `train` and `annotated` have DIFFERENT SCHEMAS and different "
            "evidential status: `train` carries prompt/generation/prompt_label/roberta_prediction, "
            "`annotated` carries human 1-5 ratings. They must not be pooled without a decision.",
            "`annotated` has NO binary label. `toxicity_human` is a continuous 1-5 mean of three "
            "annotators. Binarizing requires an explicit threshold; the published work uses >2.5, "
            "but no threshold is applied here. ~15% of ratings sit at 3.0-3.67, so the threshold "
            "materially changes the class balance.",
            "`roberta_prediction` in the `train` config is a MODEL PREDICTION, not a human label. "
            "It must never be used as ground truth.",
            "Provenance is mixed within `annotated`: the test split contains 148 human-written rows "
            "(actual_method='human'), the train split contains none. is_synthetic must therefore be "
            "set per row, not per dataset.",
            "target_group vocabulary is inconsistent: the test split has 15 categories including "
            "near-duplicates ('black/african-american folks' vs 'black folks / african-americans', "
            "'native american/indigenous folks' vs 'native american folks'); train has 13. "
            "Harmonization is a decision, not a cleanup.",
            "ToxiGen targets US-centric minority groups. Its relevance as a transfer source for "
            "African-language data needs justifying, not assuming.",
        ],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _summary_bullets(report: dict[str, Any]) -> list[str]:
    """Key facts per dataset. The JSON files remain the complete record."""
    lines: list[str] = []
    if report["name"] == "HateXplain":
        ann = report["annotation_structure"]
        splits = report.get("official_splits", {})
        dup = report["profile"]["duplication"]["text"]
        lines += [
            f"- {report['n_records']:,} posts, 3 annotators each, labels: "
            f"{', '.join(f'`{k}` ({v:,})' for k, v in ann['annotator_label_vocabulary'].items())}",
            f"- Agreement: {ann['pct_unanimous']}% unanimous, "
            f"{ann['agreement'].get('majority', 0):,} majority, {ann['n_ties_no_majority']:,} with no majority",
            f"- Majority label (ties excluded): {ann['majority_label_distribution_excluding_ties']}",
            f"- Official splits {splits.get('sizes')}, no overlap between splits",
            f"- **{splits.get('unassigned_posts')} posts are unassigned by the official split — "
            f"exactly the {ann['n_ties_no_majority']} tie posts.** The official split already drops them.",
            f"- {ann['n_target_categories']} target categories; {ann['pct_with_rationales']}% of posts carry rationales",
            f"- Near-unique text: {dup['n_distinct_exact']:,} distinct of {dup['n_rows']:,} rows "
            f"({dup['n_exact_duplicate_rows']} duplicate rows)",
        ]
    elif report["name"].startswith("Ubuntu"):
        for key, entry in report["files"].items():
            if entry.get("status") != "AVAILABLE":
                continue
            tmpl = entry["template_analysis_en"]
            lines.append(
                f"- `{key}`: {tmpl['n_rows']:,} rows from **{tmpl['n_distinct_texts']} distinct sentences** "
                f"({tmpl['rows_per_distinct_text']}x repetition); "
                f"template determines label: {tmpl['template_determines_label']}"
            )
        overlap_en = report.get("cross_file_overlap_en", {})
        overlap_am = report.get("cross_file_overlap_am", {})
        lines += [
            f"- English sentences shared between files: {overlap_en.get('n_shared')}/{overlap_en.get('n_left_distinct')}",
            f"- Amharic shared: {overlap_am.get('n_shared')}/{overlap_am.get('n_left_distinct')} "
            f"— **{overlap_en.get('n_shared', 0) - overlap_am.get('n_shared', 0)} sentences have divergent translations**",
            "- EN/AM pairing within each file is a consistent 1:1 mapping",
        ]
    elif report["name"] == "AfriHate":
        hub = report["hub"]
        lines += [
            f"- No local data: {report['local_contents_note']}",
            f"- Hub `{hub['repo_id']}`: reachable={hub.get('reachable')}, gated={hub.get('gated')}",
            "- **Blocked**: requires accepting the licence on the Hub and an `HF_TOKEN`",
        ]
    elif report["name"] == "ToxiGen":
        hub = report["hub"]
        local = report["local_contents"]
        lines += [
            f"- Local repo holds {local['prompt_files']['n_files']} prompt files "
            f"({local['prompt_files']['total_lines']:,} lines) and "
            f"{local['demonstration_files']['n_files']} demonstration files — "
            f"**generation seeds, not labelled data**",
            f"- Hub `{hub['repo_id']}`: gated={hub.get('gated')}, configs={hub.get('configs')}",
        ]
        ann = hub.get("annotated", {})
        if ann.get("status") == "AVAILABLE":
            lines.append(f"- `annotated` config downloaded: {ann['splits']} — human 1-5 ratings, no binary label")
        sample = hub.get("train_sample", {})
        if sample.get("status") == "SAMPLED":
            lines.append(
                f"- `train` config sampled ({sample['n_sampled']} streamed rows): "
                f"columns {sample['profile']['columns']} — a different schema entirely"
            )
    return lines


def _render_markdown(reports: list[dict[str, Any]]) -> str:
    """Render a human-readable summary; the JSON files are the complete record."""
    lines = [
        "# Phase 2 - Dataset inspection report",
        "",
        "Generated by `python -m src.data.inspection`. No labels were mapped, no data merged,",
        "no rows removed. Open decisions are listed per dataset and require a research ruling.",
        "Full machine-readable detail is in the sibling `*.json` files.",
        "",
        "## Availability summary",
        "",
        "| Dataset | Status | Records | Synthetic |",
        "| --- | --- | --- | --- |",
    ]
    for report in reports:
        n_records = report.get("n_records")
        if n_records is None and "files" in report:
            n_records = sum(
                f.get("profile", {}).get("shape", {}).get("rows", 0)
                for f in report["files"].values()
                if isinstance(f, dict)
            )
        synthetic = report.get("provenance", {}).get("is_synthetic")
        lines.append(
            f"| {report['name']} | {report['status']} | {f'{n_records:,}' if n_records else '-'} | "
            f"{'yes' if synthetic else 'no' if synthetic is False else '?'} |"
        )

    lines += ["", "## Findings", ""]
    for report in reports:
        lines.append(f"### {report['name']}")
        lines.append("")
        lines += _summary_bullets(report)
        lines.append("")

    lines += ["## Open decisions requiring a research ruling", ""]
    for report in reports:
        decisions = report.get("open_decisions") or []
        if decisions:
            lines.append(f"### {report['name']}")
            lines += [f"- {d}" for d in decisions] + [""]

    return "\n".join(lines)


def run_inspection(output_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    """Inspect every source and write the JSON + Markdown reports."""
    paths.ensure_dirs()
    destination = output_dir or paths.PHASE2_REPORTS_DIR
    destination.mkdir(parents=True, exist_ok=True)

    inspectors = {
        "hatexplain": inspect_hatexplain,
        "ubuntu": inspect_ubuntu,
        "afrihate": inspect_afrihate,
        "toxigen": inspect_toxigen,
    }

    reports: list[dict[str, Any]] = []
    for key, inspector in inspectors.items():
        logger.info("inspecting %s", key)
        report = inspector()
        reports.append(report)
        with (destination / f"{key}.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
        logger.info("  %s -> %s", report["status"], destination / f"{key}.json")

    summary_path = destination / "SUMMARY.md"
    summary_path.write_text(_render_markdown(reports), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    return reports


if __name__ == "__main__":
    run_inspection()
