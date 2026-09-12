"""Project paths.

Single source of truth for filesystem locations, so no module hard-codes a path
and the project relocates cleanly between a laptop and a Colab clone.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
PHASE2_REPORTS_DIR = REPORTS_DIR / "phase2_inspection"

# Source dataset locations. Third-party repos are vendored under data/raw/ but are
# git-ignored; see README.md for how each is obtained.
HATEXPLAIN_DIR = RAW_DIR / "HateXplain-master"
HATEXPLAIN_DATA = HATEXPLAIN_DIR / "Data" / "dataset.json"
HATEXPLAIN_SPLITS = HATEXPLAIN_DIR / "Data" / "post_id_divisions.json"

AFRIHATE_DIR = RAW_DIR / "AfriHate-main"
AFRIHATE_HUB_ID = "afrihate/afrihate"

TOXIGEN_DIR = RAW_DIR / "TOXIGEN-main"
TOXIGEN_HUB_ID = "toxigen/toxigen-data"

UBUNTU_MAIN_CSV = RAW_DIR / "English_Amharic_dataset_for_toxicity_detection_ubuntu_hate_speech.csv"
UBUNTU_SYNTHETIC_CSV = RAW_DIR / "ubuntu_hate_speech_synthetic_dataset_en_am_500rows.csv"


def ensure_dirs() -> None:
    """Create the output directories the pipeline writes into."""
    for directory in (PROCESSED_DIR, EMBEDDINGS_DIR, REPORTS_DIR, PHASE2_REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
