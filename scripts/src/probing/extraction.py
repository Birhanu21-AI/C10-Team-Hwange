"""Phase 8: frozen Gemma hidden-state extraction.

The model is loaded in inference mode and never fine-tuned. Hidden states are
pooled on the accelerator, moved to CPU immediately and written per layer, so GPU
memory holds only the current batch.

The sample_id -> layer -> representation mapping is preserved by writing one
array per (layer, pooling) alongside a single index of sample_ids in row order.

Run (on a GPU machine):
    python -m src.probing.extraction --split train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from src import paths
from src.data.splits import SPLIT_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIG_PATH = paths.CONFIGS_DIR / "model.yaml"


def load_model_config(path: Optional[Path] = None) -> dict[str, Any]:
    with (path or MODEL_CONFIG_PATH).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device() -> torch.device:
    """CUDA when present; CPU otherwise. MPS is skipped: unreliable for this workload."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("no CUDA device — extraction will be slow")
    return torch.device("cpu")


def pool_hidden_states(
    hidden: torch.Tensor, attention_mask: torch.Tensor, strategy: str
) -> torch.Tensor:
    """Pool one layer's token states to one vector per sequence.

    `last_token` takes the final non-padding position, which for a causal model is
    the only position that has attended to the whole sequence. `mean` averages over
    valid tokens only, so padding never dilutes the representation.
    """
    if strategy == "last_token":
        # Gemma pads on the LEFT, so the final real token is not at sum(mask)-1.
        # Locate the last position where the mask is 1, which is correct for either
        # padding side. Getting this wrong reads a padding vector and silently
        # produces meaningless representations.
        flipped = attention_mask.flip(dims=[1])
        last_index = attention_mask.shape[1] - 1 - flipped.argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
    if strategy == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling strategy: {strategy}")


def extract(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Extract pooled hidden states for every row, writing one file per layer/pooling.

    Three things keep this affordable on a free Colab GPU:

    * Results stream straight into on-disk memory-mapped arrays. Buffering every
      layer in RAM would need ~10 GB for a 43k-row corpus at 28 layers, which does
      not survive a 12.7 GB runtime alongside the model.
    * Sequences are processed in length-sorted order, so batches pad to their own
      longest member instead of to the longest in the corpus. Rows are written back
      to their original positions, so `index.parquet` order is preserved exactly.
    * Pooled vectors are cast to the output dtype on the GPU before transfer,
      halving the amount of data crossing the bus.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    extraction_config = config["extraction"]
    device = device or select_device()

    tokenizer = AutoTokenizer.from_pretrained(model_config["checkpoint"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, model_config["dtype"]) if device.type == "cuda" else torch.float32
    # AutoModel maps Gemma to Gemma3TextModel, but the checkpoint stores its tensors
    # under a `model.` prefix. Nothing matches, transformers silently returns a
    # RANDOMLY INITIALIZED network, and extraction produces a plausible-looking layer
    # curve from noise. Loading the CausalLM class matches the checkpoint layout.
    #
    # transformers 4.x and 5.x differ on both the dtype keyword and support for
    # output_loading_info, so each is attempted and degraded rather than assumed.
    load_kwargs: dict[str, Any] = {"output_hidden_states": True}
    loading_info: Optional[dict] = None
    model = None
    for dtype_kw in ("dtype", "torch_dtype"):
        try:
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                model_config["checkpoint"],
                **{dtype_kw: dtype},
                **load_kwargs,
                output_loading_info=True,
            )
            break
        except TypeError as exc:
            logger.debug("load with %s + output_loading_info failed: %s", dtype_kw, exc)
    if model is None:
        for dtype_kw in ("dtype", "torch_dtype"):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["checkpoint"], **{dtype_kw: dtype}, **load_kwargs
                )
                break
            except TypeError as exc:
                logger.debug("load with %s failed: %s", dtype_kw, exc)
    if model is None:
        raise RuntimeError(
            f"could not load {model_config['checkpoint']} with any supported dtype keyword"
        )

    # Fail loudly rather than extract from untrained weights.
    if loading_info is not None:
        missing = loading_info.get("missing_keys") or []
        if missing:
            raise RuntimeError(
                f"{len(missing)} weights were not loaded from {model_config['checkpoint']} and would "
                f"be randomly initialized (e.g. {missing[:3]}). Refusing to extract from an "
                f"untrained model."
            )
        logger.info("checkpoint loaded with all weights matched (0 missing keys)")
    else:
        logger.warning(
            "this transformers version did not return loading info; the "
            "randomly-initialized-weights guard is INACTIVE for this run"
        )

    model.eval().to(device)
    for parameter in model.parameters():  # frozen: no fine-tuning in this phase
        parameter.requires_grad_(False)

    poolings = list(extraction_config["poolings"])
    batch_size = int(model_config["batch_size"])
    max_length = int(model_config["max_length"])
    output_dtype = np.dtype(extraction_config["output_dtype"])

    texts = frame["text"].astype(str).tolist()
    n_samples = len(texts)
    hidden_size = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers) + 1  # + the embedding layer
    layer_indices = (
        list(range(n_layers))
        if extraction_config["layers"] == "all"
        else list(extraction_config["layers"])
    )
    # Validate before loading a batch: an out-of-range index otherwise fails deep in
    # the extraction loop with a bare IndexError, after the model is already loaded.
    out_of_range = [i for i in layer_indices if not 0 <= i < n_layers]
    if out_of_range:
        raise ValueError(
            f"configs/model.yaml requests layer(s) {out_of_range}, but "
            f"{model_config['checkpoint']} exposes {n_layers} (valid indices 0-{n_layers - 1}). "
            f"Set `layers: all` or choose indices within range."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    estimated_gb = n_samples * hidden_size * len(layer_indices) * len(poolings) * output_dtype.itemsize / 1024**3
    logger.info(
        "%d samples x %d layers x %d poolings x %d dims = %.1f GB on disk",
        n_samples, len(layer_indices), len(poolings), hidden_size, estimated_gb,
    )

    # One memory-mapped array per (layer, pooling); RAM stays flat regardless of corpus size.
    outputs: dict[tuple[int, str], np.memmap] = {}
    for layer_index in layer_indices:
        for pooling in poolings:
            destination = output_dir / f"layer{layer_index:02d}_{pooling}.npy"
            outputs[(layer_index, pooling)] = np.lib.format.open_memmap(
                destination, mode="w+", dtype=output_dtype, shape=(n_samples, hidden_size)
            )

    # Length-sorted order: batches pad to their own longest member, not the corpus maximum.
    lengths = [len(tokenizer(text, truncation=True, max_length=max_length)["input_ids"]) for text in texts]
    order = np.argsort(np.asarray(lengths), kind="stable")

    for start in tqdm(range(0, n_samples, batch_size), desc="extracting"):
        positions = order[start : start + batch_size]
        batch = [texts[i] for i in positions]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            hidden_states = model(**encoded).hidden_states

        for layer_index in layer_indices:
            layer = hidden_states[layer_index]
            for pooling in poolings:
                pooled = pool_hidden_states(layer, encoded["attention_mask"], pooling)
                # float32 on-device, then one transfer; rows go back to original slots.
                # Casting through float16 here would overflow Gemma's activation range
                # even when output_dtype is wider.
                vectors = pooled.to(torch.float32).cpu().numpy().astype(output_dtype)
                if not np.isfinite(vectors).all():
                    raise RuntimeError(
                        f"non-finite values in layer {layer_index} ({pooling}) at rows "
                        f"{positions[:3].tolist()}. This means the model dtype overflowed: "
                        f"use bfloat16 rather than float16 for Gemma."
                    )
                outputs[(layer_index, pooling)][positions] = vectors
        del hidden_states

    written = []
    for (layer_index, pooling), array in outputs.items():
        array.flush()
        written.append({"layer": layer_index, "pooling": pooling, "shape": list(array.shape)})
    del outputs

    index = frame[["sample_id", "label_binary", "source", "language", "split"]].reset_index(drop=True)
    index.to_parquet(output_dir / "index.parquet", index=False)

    manifest = {
        "checkpoint": model_config["checkpoint"],
        "n_samples": n_samples,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "poolings": poolings,
        "max_length": max_length,
        "device": device.type,
        "disk_gb": round(estimated_gb, 2),
        "files": written,
        "note": "row order in every .npy matches index.parquet",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("wrote %d arrays (%.1f GB) to %s", len(written), estimated_gb, output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Gemma hidden states")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "probe"])
    parser.add_argument("--limit", type=int, default=None, help="rows per split, for smoke tests")
    parser.add_argument("--checkpoint", default=None, help="override the configured checkpoint")
    parser.add_argument("--output", default=None, help="override the output directory")
    args = parser.parse_args()

    config = load_model_config()
    if args.checkpoint:
        config["model"]["checkpoint"] = args.checkpoint

    frame = pd.read_parquet(SPLIT_PATH)
    frame = frame[frame["split"].isin(args.splits) & frame["label_binary"].notna()]
    if args.limit:
        frame = frame.groupby("split", group_keys=False).head(args.limit)

    output_dir = Path(args.output) if args.output else paths.EMBEDDINGS_DIR
    extract(frame.reset_index(drop=True), config, output_dir)


if __name__ == "__main__":
    main()
