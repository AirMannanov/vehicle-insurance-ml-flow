"""CLI entry point for the MLOps pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools import get_nested, load_config, setup_logger
from src.data.bootstrap import seed_from_kaggle
from src.data.storage import load_batch
from src.analysis.data_quality import compute_batch_dq, save_batch_dq
from src.analysis.association_rules import compute_assoc_rules, save_assoc_rules
from src.analysis.dq_report import write_report
from src.database import (
    Database,
    Migrator,
    get_best_selected_model_validation_run,
    list_model_validation_runs,
    reset_project_outputs,
)
from src.models import train_models
from src.preprocessing.transformers import ensure_feature_columns
from src.reporting import write_model_report
from src.serving import load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline: streaming binary classification system"
    )
    parser.add_argument(
        "-mode",
        type=str,
        required=True,
        choices=["inference", "update", "reset", "summary", "train"],
        help="Operation mode: inference | update | reset | summary | train",
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

    def run_update(self, train_config_path: str) -> None:
        db_path = get_nested(
            self.config, "storage", "db_path", default="storage/mlops.sqlite"
        )
        with Database(db_path) as db:
            for name in Migrator(db).migrate():
                self.logger.info("Applied migration: %s", name)
            inserted = seed_from_kaggle(self.config, db)
            self._run_data_quality(db)
            self._run_assoc_rules(db)
            self._run_training(db, train_config_path=train_config_path, mode="update")
            self.logger.info(
                "Update complete: %d new batches ingested and models retrained",
                inserted,
            )

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

    def run_summary(self) -> None:
        db_path = get_nested(
            self.config, "storage", "db_path", default="storage/mlops.sqlite"
        )
        with Database(db_path) as db:
            self._ensure_no_pending_migrations(db, mode="summary")
            report_path = get_nested(
                self.config, "report", "dq_path", default="reports/dq_report.md"
            )
            path = write_report(db, report_path)
            self.logger.info("Report saved: %s", path)

    def run_inference(self, file_path: str | None) -> None:
        if file_path is None:
            self.logger.error("Inference mode requires -file argument")
            sys.exit(1)
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Inference file not found: {input_path}")
        db_path = get_nested(
            self.config, "storage", "db_path", default="storage/mlops.sqlite"
        )
        with Database(db_path) as db:
            self._ensure_no_pending_migrations(db, mode="inference")
            self.logger.info("Starting pipeline in 'inference' mode")
            record = get_best_selected_model_validation_run(db)
            bundle = load_model_bundle(record.artifact_path)
            input_df = pd.read_csv(input_path)
            output_df = self._predict_with_bundle(input_df, bundle)
            output_path = input_path.with_name(f"{input_path.stem}_predictions.csv")
            output_df.to_csv(output_path, index=False)
            self.logger.info(
                "Inference complete: model=%s validation_run_id=%s output=%s",
                record.model_name,
                record.validation_run_id,
                output_path,
            )

    def _predict_with_bundle(
        self,
        df: pd.DataFrame,
        bundle: dict[str, Any],
    ) -> pd.DataFrame:
        model_name = bundle["model_name"]
        model = bundle["model"]
        feature_spec = bundle.get("feature_spec", {})
        training_metadata = bundle.get("training_metadata", {})
        threshold = float(training_metadata.get("threshold", 0.5))

        feature_columns = list(feature_spec.get("feature_columns", []))
        numeric_features = list(feature_spec.get("numeric_features", []))
        categorical_features = list(feature_spec.get("categorical_features", []))
        dropped_columns = list(feature_spec.get("dropped_columns", []))

        prepared_df = df.drop(columns=dropped_columns, errors="ignore").copy()
        prepared_df = ensure_feature_columns(
            prepared_df,
            numeric_features,
            categorical_features,
        )

        if model_name == "mlp":
            preprocessor = bundle.get("preprocessor")
            if preprocessor is None:
                raise RuntimeError("MLP inference requires a fitted preprocessor in the model bundle")
            X = preprocessor.transform(prepared_df)
        elif model_name == "catboost":
            X = prepared_df[feature_columns].copy()
            for column in categorical_features:
                if column not in X.columns:
                    continue
                X[column] = X[column].where(X[column].notna(), "missing").astype(str)
        else:
            raise ValueError(f"Unsupported model_name in artifact bundle: {model_name!r}")

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        else:
            probabilities = np.asarray(model.predict(X), dtype=float)

        predictions = (probabilities >= threshold).astype(int)
        output_df = df.copy()
        output_df["predict_proba"] = probabilities
        output_df["predict"] = predictions
        return output_df

    def run_train(self, train_config_path: str) -> None:
        self.logger.info("Starting pipeline in 'train' mode")
        db_path = get_nested(self.config, "storage", "db_path")
        with Database(db_path) as db:
            self._run_training(db, train_config_path=train_config_path, mode="train")

    def _run_training(
        self,
        db: Database,
        *,
        train_config_path: str,
        mode: str,
    ) -> None:
        self._ensure_no_pending_migrations(db, mode=mode)
        train_config = load_config(train_config_path)
        results = train_models(db, self.config, train_config)
        for result in results:
            self.logger.info(
                "%s complete: model=%s validation_run_id=%s train_rows=%s selected=%s artifact=%s",
                mode.capitalize(),
                result.model_name,
                result.validation_run_id,
                result.train_rows,
                result.is_selected,
                result.artifact_path,
            )
            record = next(
                record
                for record in list_model_validation_runs(db, model_name=result.model_name)
                if record.validation_run_id == result.validation_run_id
            )
            report_path = write_model_report(record)
            self.logger.info("Model report saved: %s", report_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(config)
    runner = PipelineRunner(config, logger)

    if args.mode == "reset":
        runner.run_reset(args.train_config)
        return
    if args.mode == "update":
        runner.run_update(args.train_config)
        return
    if args.mode == "summary":
        runner.run_summary()
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
