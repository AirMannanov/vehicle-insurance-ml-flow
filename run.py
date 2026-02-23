"""CLI entry point for the MLOps pipeline."""

import argparse
import sys

from src.config import load_config, get_nested
from src.database import Database, Migrator
from src.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline: streaming binary classification system"
    )
    parser.add_argument(
        "-mode",
        type=str,
        required=True,
        choices=["inference", "update", "summary"],
        help="Operation mode: inference | update | summary",
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


def run_update(config: dict, db: Database) -> None:
    from src.data.loader import download_dataset, load_all_csv
    from src.data.batch_generator import generate_batches
    from src.data.storage import save_all_batches

    import logging
    logger = logging.getLogger("mlops")

    raw_dir = get_nested(config, "data", "raw_dir", default="data")
    time_col = get_nested(config, "data", "time_column", default="INSR_BEGIN")

    data_dir = download_dataset(dest_dir=raw_dir)
    df = load_all_csv(data_dir)

    batches = generate_batches(df, time_column=time_col)
    inserted = save_all_batches(db, batches)

    logger.info("Update complete: %d new batches ingested", inserted)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config)

    db_path = get_nested(config, "storage", "db_path", default="storage/mlops.sqlite")
    db = Database(db_path)
    applied = Migrator(db).migrate()
    for name in applied:
        logger.info(f"Applied migration: {name}")

    logger.info(f"Starting pipeline in '{args.mode}' mode")

    if args.mode == "inference":
        if args.file is None:
            logger.error("Inference mode requires -file argument")
            sys.exit(1)
        raise NotImplementedError("Inference mode not yet implemented")

    elif args.mode == "update":
        run_update(config, db)

    elif args.mode == "summary":
        raise NotImplementedError("Summary mode not yet implemented")

    db.close()


if __name__ == "__main__":
    main()
