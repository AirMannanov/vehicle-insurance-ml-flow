import logging
import sys
from pathlib import Path
from typing import Any

from src.tools.config import get_nested


def setup_logger(
    config: dict[str, Any],
    name: str = "mlops",
) -> logging.Logger:
    """Configure and return a logger with console and file handlers.

    Args:
        config: Pipeline configuration dictionary.
        name: Logger name.

    Returns:
        Configured Logger instance.
    """
    level_str = get_nested(config, "logging", "level", default="INFO")
    log_file = get_nested(config, "logging", "log_file", default="logs/mlops.log")
    level = getattr(logging, level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
