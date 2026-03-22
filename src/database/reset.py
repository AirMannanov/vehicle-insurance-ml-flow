import logging
import shutil
from pathlib import Path
from typing import Any

from src.tools.config import get_nested, load_config

logger = logging.getLogger("mlops")

_DEFAULT_TRAIN_CONFIG_PATH = "train_config.yaml"


def reset_db(config: dict[str, Any] | None = None, db_path: str | None = None) -> list[str]:
    """Delete SQLite DB files. Next run will create a new DB and re-seed."""
    path_str = db_path or get_nested(
        config or {},
        "storage",
        "db_path",
        default="storage/mlops.sqlite",
    )
    path = Path(path_str)
    removed: list[str] = []
    for db_file in (path, path.parent / f"{path.name}-wal", path.parent / f"{path.name}-shm"):
        if db_file.exists():
            db_file.unlink()
            removed.append(str(db_file))
    if removed:
        logger.info("Removed DB files: %s", ", ".join(removed))
    else:
        logger.info("No DB file found at %s", path_str)
    return removed


def reset_project_outputs(
    config: dict[str, Any],
    *,
    train_config_path: str = _DEFAULT_TRAIN_CONFIG_PATH,
) -> list[str]:
    removed_paths: list[str] = []
    removed_paths.extend(reset_db(config))

    train_config = _load_optional_train_config(train_config_path)
    for path in _paths_to_clean(config, train_config=train_config):
        removed_paths.extend(_clear_path(path))

    if removed_paths:
        logger.info("Reset removed project outputs: %s", ", ".join(removed_paths))
    else:
        logger.info("Reset found no project outputs to remove")
    return removed_paths


def _load_optional_train_config(train_config_path: str) -> dict[str, Any]:
    path = Path(train_config_path)
    if not path.exists():
        return {}
    return load_config(str(path))


def _paths_to_clean(
    config: dict[str, Any],
    *,
    train_config: dict[str, Any],
) -> list[Path]:
    raw_paths = [
        get_nested(config, "logging", "log_file", default="logs/mlops.log"),
        get_nested(config, "report", "dq_path", default="reports/dq_report.md"),
        get_nested(train_config, "output", "artifacts_dir", default="artifacts/models"),
        "catboost_info",
        "__pycache__",
    ]

    paths: list[Path] = []
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        paths.append(Path(raw_path))

    report_dir = Path(get_nested(config, "report", "dq_path", default="reports/dq_report.md")).parent
    logs_dir = Path(get_nested(config, "logging", "log_file", default="logs/mlops.log")).parent
    for extra_dir in (report_dir, logs_dir, Path("artifacts")):
        if extra_dir not in paths:
            paths.append(extra_dir)
    return paths


def _clear_path(path: Path) -> list[str]:
    removed: list[str] = []
    if not path.exists():
        return removed

    if path.is_file():
        path.unlink()
        removed.append(str(path))
        return removed

    for child in sorted(path.iterdir()):
        if child.name in {".gitkeep"}:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(str(child))
    return removed
