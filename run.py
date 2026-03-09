"""CLI entry point for the MLOps pipeline."""

import argparse
import sys

from src.tools import load_config, get_nested, setup_logger
from src.data.bootstrap import seed_from_kaggle
from src.database import Database, Migrator, reset_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline: streaming binary classification system"
    )
    parser.add_argument(
        "-mode",
        type=str,
        required=True,
        choices=["inference", "update", "reset-db"],
        help="Operation mode: inference | update | reset-db",
    )
    parser.add_argument(
        "-file",
        type=str,
        default=None,
        help="Path to input data file (required for inference mode)",
    )
    parser.add_argument(
        "-config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config)

    if args.mode == "reset-db":
        reset_db(config)
        return

    db_path = get_nested(config, "storage", "db_path", default="storage/mlops.sqlite")
    db = Database(db_path)

    if args.mode == "update":
        for name in Migrator(db).migrate():
            logger.info("Applied migration: %s", name)
        inserted = seed_from_kaggle(config, db)
        logger.info("Update complete: %d new batches ingested", inserted)
        db.close()
        return

    logger.info("Starting pipeline in '%s' mode", args.mode)
    if args.mode == "inference":
        if args.file is None:
            logger.error("Inference mode requires -file argument")
            sys.exit(1)
        raise NotImplementedError("Inference mode not yet implemented")

    db.close()


if __name__ == "__main__":
    main()
