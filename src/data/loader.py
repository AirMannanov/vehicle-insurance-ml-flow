import logging
import shutil
from pathlib import Path

import kagglehub
import pandas as pd

logger = logging.getLogger("mlops")

KAGGLE_DATASET = "imtkaggleteam/vehicle-insurance-data"


def download_dataset(dest_dir: str = "data", force_download: bool = False) -> Path:
    """Download dataset via kagglehub and copy CSV files to dest_dir.

    Returns the directory path containing the CSV files.
    If CSV files already exist in dest_dir, the download is skipped
    unless force_download=True (use for temporary directories).
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    existing = list(dest.glob("*.csv"))
    if existing and not force_download:
        logger.info("CSV already present in %s, skipping download", dest)
        return dest

    logger.info("Downloading dataset %s via kagglehub …", KAGGLE_DATASET)
    src_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    logger.info("Downloaded to cache: %s", src_path)

    csv_files = list(src_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {src_path}")

    for csv in csv_files:
        target = dest / csv.name
        shutil.copy2(csv, target)
        logger.info("Copied %s → %s", csv.name, target)

    return dest


def load_all_csv(data_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all CSV files from *data_dir*."""
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows × %d columns", path.name, len(df), len(df.columns))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined: %d rows × %d columns from %d files", len(combined), len(combined.columns), len(frames))
    return combined
