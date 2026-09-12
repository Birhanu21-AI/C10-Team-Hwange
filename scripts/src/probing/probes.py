"""Phases 9-10: layer-wise linear probes and their controls.

A probe is deliberately weak -- StandardScaler + LogisticRegression -- so that
performance reflects what the representation makes linearly available, not what a
powerful classifier can learn on top of it.

Controls (Phase 10) exist because a high probe score alone proves nothing:
  * shuffled labels  -> the ceiling reachable from label noise and memorization
  * random embeddings -> what the pipeline scores on structureless input
  * pooling variants  -> whether results depend on how tokens were pooled
A layer's result is only interesting relative to these floors and to the Phase 7
TF-IDF baseline.

Run:
    python -m src.probing.probes --embeddings data/embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import paths
from src.analysis.metrics import classification_metrics
from src.probing.extraction import MODEL_CONFIG_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = paths.REPORTS_DIR / "phase9_probes" / "probe_results.json"


def _build_probe(config: dict[str, Any], seed: int) -> Pipeline:
    steps = []
    if config.get("standardize", True):
        steps.append(("scale", StandardScaler()))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=int(config.get("max_iter", 2000)),
                class_weight=config.get("class_weight", "balanced"),
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def _fit_and_score(
    features: np.ndarray,
    index: pd.DataFrame,
    probe_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Train on the train split and evaluate on every other split present."""
    train_mask = (index["split"] == "train").to_numpy()
    if train_mask.sum() == 0:
        return {"error": "no training rows"}

    # Surface a corrupted extraction as itself, not as an opaque sklearn error.
    if not np.isfinite(np.asarray(features[train_mask], dtype=np.float32)).all():
        return {
            "error": "features contain inf or nan — re-extract with model dtype bfloat16, "
                     "not float16 (Gemma activations exceed the float16 range)"
        }

    y_train = index.loc[train_mask, "label_binary"].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        return {"error": "training rows contain a single class"}

    model = _build_probe(probe_config, seed)
    model.fit(features[train_mask].astype(np.float32), y_train)

    results: dict[str, Any] = {}
    for split_name in index["split"].dropna().unique():
        if split_name == "train":
            continue
        mask = (index["split"] == split_name).to_numpy()
        if mask.sum() == 0:
            continue
        subset = features[mask].astype(np.float32)
        y_true = index.loc[mask, "label_binary"].to_numpy(dtype=int)
        results[str(split_name)] = classification_metrics(
            y_true, model.predict(subset), model.predict_proba(subset)[:, 1]
        )
        # per-language breakdown, reported separately rather than pooled
        languages = index.loc[mask, "language"]
        if languages.nunique() > 1:
            by_language = {}
            for language in languages.unique():
                lang_mask = (languages == language).to_numpy()
                by_language[str(language)] = classification_metrics(
                    y_true[lang_mask],
                    model.predict(subset[lang_mask]),
                    model.predict_proba(subset[lang_mask])[:, 1],
                )
            results[str(split_name)]["by_language"] = by_language
    return results


def run_probes(embeddings_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Probe every extracted layer/pooling, then run the Phase 10 controls."""
    with MODEL_CONFIG_PATH.open(encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)
    probe_config = model_config["probing"]
    control_config = probe_config.get("controls", {})

    # An empty or missing embeddings directory means extraction has not been run
    # since it was last cleared -- say that, rather than failing on a missing file.
    if not (embeddings_dir / "index.parquet").exists():
        raise SystemExit(
            f"No embeddings in {embeddings_dir}. Run extraction first:\n"
            f"  python -m src.probing.extraction --splits train val test probe"
        )

    index = pd.read_parquet(embeddings_dir / "index.parquet")
    with (embeddings_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    results: dict[str, Any] = {
        "checkpoint": manifest["checkpoint"],
        "n_samples": manifest["n_samples"],
        "seed": seed,
        "layers": {},
        "controls": {},
    }

    rng = np.random.default_rng(seed)
    for entry in sorted(manifest["files"], key=lambda e: (e["layer"], e["pooling"])):
        layer, pooling = entry["layer"], entry["pooling"]
        # mmap_mode keeps the full array on disk; only the rows a split needs are
        # materialized, so probing 56 layer/pooling combinations never accumulates.
        features = np.load(embeddings_dir / f"layer{layer:02d}_{pooling}.npy", mmap_mode="r")
        key = f"layer{layer:02d}/{pooling}"
        results["layers"][key] = _fit_and_score(features, index, probe_config, seed)
        logger.info("probed %s", key)

    # ---- Phase 10 controls, run on the final layer's representation ----
    last = sorted(manifest["files"], key=lambda e: e["layer"])[-1]
    features = np.load(
        embeddings_dir / f"layer{last['layer']:02d}_{last['pooling']}.npy", mmap_mode="r"
    )

    if control_config.get("shuffled_labels"):
        shuffled_index = index.copy()
        permuted = shuffled_index["label_binary"].to_numpy().copy()
        rng.shuffle(permuted)
        shuffled_index["label_binary"] = permuted
        results["controls"]["shuffled_labels"] = {
            "on": f"layer{last['layer']:02d}/{last['pooling']}",
            "expectation": "near chance; anything higher indicates leakage or memorization",
            "result": _fit_and_score(features, shuffled_index, probe_config, seed),
        }

    if control_config.get("random_embeddings"):
        random_features = rng.normal(size=features.shape).astype(np.float32)
        results["controls"]["random_embeddings"] = {
            "shape": list(features.shape),
            "expectation": "near chance; the pipeline itself must not produce signal",
            "result": _fit_and_score(random_features, index, probe_config, seed),
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    logger.info("wrote %s", REPORT_PATH)
    return results


def layer_curve(results: dict[str, Any], split: str = "test", metric: str = "f1") -> pd.DataFrame:
    """Layer-vs-performance table, the primary Phase 9 output."""
    rows = []
    for key, per_split in results["layers"].items():
        if "error" in per_split or split not in per_split:
            continue
        layer, pooling = key.split("/")
        rows.append(
            {
                "layer": int(layer.replace("layer", "")),
                "pooling": pooling,
                metric: per_split[split].get(metric),
                "roc_auc": per_split[split].get("roc_auc"),
            }
        )
    columns = ["layer", "pooling", metric, "roc_auc"]
    if not rows:
        logger.warning("no probe results for split %r — nothing to plot", split)
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["pooling", "layer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise linear probes")
    parser.add_argument("--embeddings", default=str(paths.EMBEDDINGS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_probes(Path(args.embeddings), seed=args.seed)
    print(layer_curve(results).to_string(index=False))


if __name__ == "__main__":
    main()
