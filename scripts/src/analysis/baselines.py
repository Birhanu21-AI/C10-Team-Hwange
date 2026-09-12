"""Phase 7: TF-IDF + Logistic Regression baselines.

These establish what surface lexical features achieve before any Gemma
representation is involved. Phase 10 reuses them as a probe control: a probe that
does not beat TF-IDF has not demonstrated that the model encodes anything extra.

Run:
    python -m src.analysis.baselines
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config as config_module
from src import paths
from src.analysis.metrics import aggregate_seeds, classification_metrics
from src.data.splits import SPLIT_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = paths.REPORTS_DIR / "phase7_baselines" / "baseline_results.json"

# Word features suit English; character n-grams are needed for Amharic (Ge'ez script),
# so both are reported rather than one being assumed adequate for a bilingual corpus.
VECTORIZERS = {
    "word_1_2": dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=200_000),
    "char_wb_2_5": dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200_000),
}

SEEDS = [42, 43, 44]

# Impact Assessment Card 7: "Keyword shortcuts -> test examples with paraphrasing and
# different wording." Neutral group names are masked in the test text and the same
# trained model is re-scored. A large drop means the model keys on the mention of a
# group rather than on harmful meaning -- the keyword-shortcut risk both cards name.
IDENTITY_TERMS = [
    "african", "arab", "asian", "caucasian", "hispanic", "indian", "indigenous",
    "jewish", "jew", "jews", "refugee", "refugees", "minority", "muslim", "muslims",
    "islam", "islamic", "black", "white", "mexican", "chinese", "latino", "latina",
    "women", "woman", "men", "man", "gay", "lesbian", "homosexual", "trans",
    "immigrant", "immigrants", "christian", "hindu", "buddhist",
]
IDENTITY_PATTERN = re.compile(r"\b(" + "|".join(IDENTITY_TERMS) + r")\b", flags=re.IGNORECASE)
MASK_TOKEN = "person"

# Honest limitation: LogisticRegression with the lbfgs solver on fixed data is
# deterministic, so `random_state` does not perturb it and the reported std is
# always 0.0. This is recorded rather than presented as stability evidence.
# Phase 14 must obtain variance from a real source (bootstrap resampling of the
# training set, or resampled group-level splits), not from a seed that does nothing.
SEED_VARIANCE_NOTE = (
    "std is 0.0 by construction: lbfgs LogisticRegression is deterministic on fixed "
    "data, so random_state has no effect. Not evidence of stability."
)


def _build_pipeline(vectorizer_kwargs: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]
    )


def _evaluate_slices(frame: pd.DataFrame, y_pred, y_score, by: str) -> dict[str, Any]:
    """Metrics broken down by a column, reported separately rather than pooled."""
    results = {}
    working = frame.assign(_pred=y_pred, _score=y_score)
    for value, group in working.groupby(by):
        if group[by].isna().all():
            continue
        results[str(value)] = classification_metrics(
            group["label_binary"].to_numpy(dtype=int),
            group["_pred"].to_numpy(dtype=int),
            group["_score"].to_numpy(dtype=float),
        )
    return results


def _mask_identity_terms(texts: pd.Series) -> pd.Series:
    """Replace neutral group names with a placeholder, leaving everything else intact."""
    return texts.astype(str).str.replace(IDENTITY_PATTERN, MASK_TOKEN, regex=True)


def _data_card_slices(frame: pd.DataFrame, y_pred, y_score) -> dict[str, Any]:
    """Evaluation slices named by the Data Card (6, 7) and Impact Card (8.1)."""
    working = frame.assign(_pred=y_pred, _score=y_score)
    slices: dict[str, Any] = {}

    ethnic = working[working["is_ethnic_target"] == True]  # noqa: E712
    if not ethnic.empty:
        slices["ethnic_target_all"] = classification_metrics(
            ethnic["label_binary"].to_numpy(dtype=int),
            ethnic["_pred"].to_numpy(dtype=int),
            ethnic["_score"].to_numpy(dtype=float),
        )

    # Data Card 7: false-positive check on legitimate ethnic-group discussion.
    legitimate = ethnic[ethnic["label_binary"] == 0]
    if not legitimate.empty:
        n_flagged = int((legitimate["_pred"] == 1).sum())
        slices["legitimate_ethnic_discussion"] = {
            "n": int(len(legitimate)),
            "n_incorrectly_flagged_toxic": n_flagged,
            "false_positive_rate": round(n_flagged / len(legitimate), 4),
            "note": "non-toxic rows that name an ethnic group; flagging these suppresses legitimate speech",
        }

    for column in ("toxicity_type", "harm_type"):
        if column in working and working[column].notna().any():
            slices[f"by_{column}"] = {}
            for value, group in working[working[column].notna()].groupby(column):
                slices[f"by_{column}"][str(value)] = classification_metrics(
                    group["label_binary"].to_numpy(dtype=int),
                    group["_pred"].to_numpy(dtype=int),
                    group["_score"].to_numpy(dtype=float),
                )
    return slices


def run_baselines() -> dict[str, Any]:
    """Train on `train`, evaluate on `test` and on the held-out Ubuntu probe set."""
    settings = config_module.load_config()
    frame = pd.read_parquet(SPLIT_PATH)
    frame = frame[frame["label_binary"].notna()]

    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    probe = frame[frame["split"] == "probe"]

    results: dict[str, Any] = {
        "policy": config_module.policy_fingerprint(settings),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_probe": int(len(probe)),
        "seeds": SEEDS,
        "seed_variance_note": SEED_VARIANCE_NOTE,
        "vectorizers": {},
    }

    for name, kwargs in VECTORIZERS.items():
        logger.info("baseline: %s", name)
        per_seed_test: list[dict[str, Any]] = []
        per_seed_probe: list[dict[str, Any]] = []
        last_model = None

        for seed in SEEDS:
            model = _build_pipeline(kwargs, seed)
            model.fit(train["text"], train["label_binary"].astype(int))
            last_model = model

            for subset, collector in ((test, per_seed_test), (probe, per_seed_probe)):
                if subset.empty:
                    continue
                pred = model.predict(subset["text"])
                score = model.predict_proba(subset["text"])[:, 1]
                collector.append(
                    classification_metrics(
                        subset["label_binary"].to_numpy(dtype=int), pred, score
                    )
                )

        entry: dict[str, Any] = {
            "test": aggregate_seeds(per_seed_test),
            "test_single_seed_detail": per_seed_test[0],
        }
        if per_seed_probe:
            entry["ubuntu_probe_out_of_domain"] = aggregate_seeds(per_seed_probe)

        # Breakdowns use the final seed's model; slices are reported separately.
        test_pred = last_model.predict(test["text"])
        test_score = last_model.predict_proba(test["text"])[:, 1]
        entry["by_source"] = _evaluate_slices(test, test_pred, test_score, "source")
        entry["by_language"] = _evaluate_slices(test, test_pred, test_score, "language")
        entry["data_card_slices"] = _data_card_slices(test, test_pred, test_score)

        # Keyword-shortcut control: same model, identity terms masked.
        masked_text = _mask_identity_terms(test["text"])
        masked_pred = last_model.predict(masked_text)
        masked_score = last_model.predict_proba(masked_text)[:, 1]
        masked_metrics = classification_metrics(
            test["label_binary"].to_numpy(dtype=int), masked_pred, masked_score
        )
        entry["keyword_shortcut_control"] = {
            "masked_identity_terms": masked_metrics,
            "f1_drop": round(entry["test_single_seed_detail"]["f1"] - masked_metrics["f1"], 4),
            "n_texts_changed": int((masked_text != test["text"]).sum()),
            "interpretation": "a large F1 drop indicates reliance on group mentions rather than harmful meaning",
        }

        if not probe.empty:
            probe_pred = last_model.predict(probe["text"])
            probe_score = last_model.predict_proba(probe["text"])[:, 1]
            entry["probe_by_language"] = _evaluate_slices(probe, probe_pred, probe_score, "language")
            entry["probe_by_harm_type"] = _evaluate_slices(probe, probe_pred, probe_score, "harm_type")

        results["vectorizers"][name] = entry

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info("wrote %s", REPORT_PATH)
    return results


if __name__ == "__main__":
    output = run_baselines()
    for vec, entry in output["vectorizers"].items():
        print(f"\n== {vec} ==")
        print("  test          :", {k: v for k, v in entry["test"].items() if k in ("f1", "balanced_accuracy", "roc_auc")})
        if "ubuntu_probe_out_of_domain" in entry:
            print("  ubuntu probe  :", {k: v for k, v in entry["ubuntu_probe_out_of_domain"].items() if k in ("f1", "balanced_accuracy", "roc_auc")})
        print("  by source     :", {s: m["f1"] for s, m in entry["by_source"].items()})
        print("  by language   :", {s: m["f1"] for s, m in entry["by_language"].items()})
