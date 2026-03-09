"""CLI entry point for the MLOps pipeline."""

import argparse
import logging
import sys
from typing import Any

from src.tools import get_nested, load_config, setup_logger
from src.data.bootstrap import seed_from_kaggle
from src.data.storage import load_batch
from src.analysis.data_quality import compute_batch_dq, save_batch_dq
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


class PipelineRunner:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run_reset_db(self) -> None:
        reset_db(self.config)

    def run_update(self) -> None:
        db_path = get_nested(
            self.config, "storage", "db_path", default="storage/mlops.sqlite"
        )
        db = Database(db_path)
        try:
            for name in Migrator(db).migrate():
                self.logger.info("Applied migration: %s", name)
            inserted = seed_from_kaggle(self.config, db)
            self._run_data_quality(db)
            self.logger.info("Update complete: %d new batches ingested", inserted)
        finally:
            db.close()

    def _run_data_quality(self, db: Database) -> None:
        batches_without_dq = db.fetchall(
            """
            SELECT batch_id 
            FROM raw_batches 
            WHERE batch_id NOT IN (
                SELECT DISTINCT batch_id 
                FROM data_quality
            )"""
        )
        for row in batches_without_dq:
            batch_id = row["batch_id"]
            df = load_batch(db, batch_id)
            dq_rows = compute_batch_dq(df)
            save_batch_dq(db, batch_id, dq_rows)
            self.logger.info("Computed DQ for batch %s", batch_id)
        if batches_without_dq:
            self.logger.info("Data quality: %d batches updated", len(batches_without_dq))

    def run_inference(self, file_path: str | None) -> None:
        if file_path is None:
            self.logger.error("Inference mode requires -file argument")
            sys.exit(1)
        db_path = get_nested(
            self.config, "storage", "db_path", default="storage/mlops.sqlite"
        )
        db = Database(db_path)
        try:
            self.logger.info("Starting pipeline in 'inference' mode")
            raise NotImplementedError("Inference mode not yet implemented")
        finally:
            db.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config)
    runner = PipelineRunner(config, logger)

    if args.mode == "reset-db":
        runner.run_reset_db()
        return
    if args.mode == "update":
        runner.run_update()
        return
    if args.mode == "inference":
        runner.run_inference(args.file)
        return

    logger.error("Unexpected mode: %s", args.mode)
    sys.exit(1)


if __name__ == "__main__":
    main()
