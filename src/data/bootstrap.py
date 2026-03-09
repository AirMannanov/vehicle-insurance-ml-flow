import logging
import tempfile
from typing import Any

from src.tools.config import get_nested
from src.database.connection import Database

logger = logging.getLogger("mlops")


def seed_from_kaggle(config: dict[str, Any], db: Database) -> int:
    """Download dataset to a temporary directory, load into SQLite, then delete temp dir.

    Returns number of newly inserted batches.
    """
    from src.data.loader import download_dataset, load_all_csv
    from src.data.batch_generator import generate_batches
    from src.data.storage import save_all_batches

    time_col = get_nested(config, "data", "time_column", default="INSR_BEGIN")

    with tempfile.TemporaryDirectory(prefix="mlops_fetch_") as tmp_dir:
        logger.info("Downloading data to temporary directory %s", tmp_dir)
        download_dataset(dest_dir=tmp_dir, force_download=True)
        df = load_all_csv(tmp_dir)
        batches = generate_batches(df, time_column=time_col)
        inserted = save_all_batches(db, batches)
    return inserted
