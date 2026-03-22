"""CLI entry point for the MLOps pipeline."""

import argparse
import logging
import sys
from typing import Any

from src.tools import get_nested, load_config, setup_logger
from src.data.bootstrap import seed_from_kaggle
from src.data.storage import load_batch
from src.analysis.data_quality import compute_batch_dq, save_batch_dq
from src.analysis.association_rules import compute_assoc_rules, save_assoc_rules
from src.analysis.dq_report import write_report
from src.database import Database, Migrator, reset_project_outputs
from src.models import train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline: streaming binary classification system"
    )
    parser.add_argument(
        "-mode",
        type=str,
        required=True,
        choices=["inference", "update", "reset", "report", "train"],
        help="Operation mode: inference | update | reset | report | train",
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
    parser.add_argument(
        "-train-config",
        type=str,
        default="train_config.yaml",
        help="Path to train configuration file (used in train mode)",
    )
    return parser.parse_args()


class PipelineRunner:
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def _ensure_no_pending_migrations(self, db: Database, *, mode: str) -> None:
        migration_status = Migrator(db).status()
        pending_migrations = [
            name for name, is_applied in migration_status.items() if not is_applied
        ]
        if pending_migrations:
            raise RuntimeError(
                f"{mode.capitalize()} mode requires an up-to-date database schema. "
                f"Pending migrations: {pending_migrations}"
            )

    def run_reset(self, train_config_path: str) -> None:
        reset_project_outputs(self.config, train_config_path=train_config_path)

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
            self._run_assoc_rules(db)
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

    def _run_assoc_rules(self, db: Database) -> None:
        exclude = {
            get_nested(self.config, "data", "time_column", default="INSR_BEGIN"),
            get_nested(self.config, "data", "target_column", default="CLAIM_PAID"),
        }
        batches_without_rules = db.fetchall(
            """
            SELECT batch_id
            FROM raw_batches
            WHERE batch_id NOT IN (
                SELECT DISTINCT batch_id
                FROM assoc_rules
            )
            """
        )
        for row in batches_without_rules:
            batch_id = row["batch_id"]
            df = load_batch(db, batch_id)
            rules = compute_assoc_rules(df, exclude_columns=exclude)
            save_assoc_rules(db, batch_id, rules)
            self.logger.info("Computed association rules for batch %s: %d rules", batch_id, len(rules))
        if batches_without_rules:
            self.logger.info(
                "Association rules: %d batches updated", len(batches_without_rules)
            )

    def run_report(self) -> None:
        db_path = get_nested(
            self.config, "storage", "db_path", default="storage/mlops.sqlite"
        )
        db = Database(db_path)
        try:
            report_path = get_nested(
                self.config, "report", "dq_path", default="reports/dq_report.md"
            )
            path = write_report(db, report_path)
            self.logger.info("Report saved: %s", path)
        finally:
            db.close()

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

    def run_train(self, train_config_path: str) -> None:
        self.logger.info("Starting pipeline in 'train' mode")
        train_config = load_config(train_config_path)
        db_path = get_nested(self.config, "storage", "db_path")
        db = Database(db_path)
        try:
            self._ensure_no_pending_migrations(db, mode="train")
            results = train_models(db, self.config, train_config)
            for result in results:
                self.logger.info(
                    "Train complete: model=%s validation_run_id=%s train_rows=%s selected=%s artifact=%s",
                    result.model_name,
                    result.validation_run_id,
                    result.train_rows,
                    result.is_selected,
                    result.artifact_path,
                )
        finally:
            db.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config)
    runner = PipelineRunner(config, logger)

    if args.mode == "reset":
        runner.run_reset(args.train_config)
        return
    if args.mode == "update":
        runner.run_update()
        return
    if args.mode == "report":
        runner.run_report()
        return
    if args.mode == "inference":
        runner.run_inference(args.file)
        return
    if args.mode == "train":
        runner.run_train(args.train_config)
        return

    logger.error("Unexpected mode: %s", args.mode)
    sys.exit(1)


if __name__ == "__main__":
    main()
