import logging
from pathlib import Path
from typing import Any

from src.tools.config import get_nested

logger = logging.getLogger("mlops")


def reset_db(config: dict[str, Any] | None = None, db_path: str | None = None) -> list[str]:
    """Delete SQLite DB files. Next run will create a new DB and re-seed"""
    path_str = db_path or get_nested(config or {}, "storage", "db_path", default="storage/mlops.sqlite")
    path = Path(path_str)
    removed = []
    for p in (path, path.parent / (path.name + "-wal"), path.parent / (path.name + "-shm")):
        if p.exists():
            p.unlink()
            removed.append(str(p))
    if removed:
        logger.info("Removed DB files: %s", ", ".join(removed))
    else:
        logger.info("No DB file found at %s", path_str)
    logger.info("Next run will create a new DB and seed from Kaggle.")
    return removed
