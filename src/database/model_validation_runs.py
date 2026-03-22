import json
from dataclasses import dataclass
from typing import Any

from src.database.connection import Database


@dataclass(slots=True)
class ModelValidationRunRecord:
    validation_run_id: int
    model_name: str
    artifact_path: str
    hyperparameters_json: str
    feature_spec_json: str
    validation_metrics_json: str
    test_metrics_json: str | None
    split_config_json: str
    is_selected: int
    created_at: str


def insert_model_validation_run(
    db: Database,
    *,
    model_name: str,
    artifact_path: str,
    hyperparameters: dict[str, Any],
    feature_spec: dict[str, Any],
    validation_metrics: dict[str, Any],
    split_config: dict[str, Any],
) -> int:
    cursor = db.execute(
        """
        INSERT INTO model_validation_runs (
            model_name,
            artifact_path,
            hyperparameters_json,
            feature_spec_json,
            validation_metrics_json,
            split_config_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            model_name,
            artifact_path,
            json.dumps(hyperparameters, sort_keys=True),
            json.dumps(feature_spec, sort_keys=True),
            json.dumps(validation_metrics, sort_keys=True),
            json.dumps(split_config, sort_keys=True),
        ),
    )
    db.commit()
    return int(cursor.lastrowid)


def update_model_validation_run_test_metrics(
    db: Database,
    *,
    validation_run_id: int,
    test_metrics: dict[str, Any],
) -> None:
    db.execute(
        """
        UPDATE model_validation_runs
        SET test_metrics_json = ?
        WHERE validation_run_id = ?
        """,
        (json.dumps(test_metrics, sort_keys=True), validation_run_id),
    )
    db.commit()


def clear_selected_model_for_name(
    db: Database,
    *,
    model_name: str,
) -> None:
    db.execute(
        """
        UPDATE model_validation_runs
        SET is_selected = 0
        WHERE model_name = ?
        """,
        (model_name,),
    )
    db.commit()


def mark_model_validation_run_selected(
    db: Database,
    *,
    validation_run_id: int,
) -> None:
    db.execute(
        """
        UPDATE model_validation_runs
        SET is_selected = 1
        WHERE validation_run_id = ?
        """,
        (validation_run_id,),
    )
    db.commit()


def list_model_validation_runs(
    db: Database,
    *,
    model_name: str | None = None,
) -> list[ModelValidationRunRecord]:
    params: list[Any] = []
    where_clause = ""
    if model_name is not None:
        where_clause = "WHERE model_name = ?"
        params.append(model_name)

    rows = db.fetchall(
        f"""
        SELECT
            validation_run_id,
            model_name,
            artifact_path,
            hyperparameters_json,
            feature_spec_json,
            validation_metrics_json,
            test_metrics_json,
            split_config_json,
            is_selected,
            created_at
        FROM model_validation_runs
        {where_clause}
        ORDER BY validation_run_id ASC
        """,
        params,
    )
    return [ModelValidationRunRecord(**row) for row in rows]


def get_selected_model_validation_run(
    db: Database,
    *,
    model_name: str,
) -> ModelValidationRunRecord | None:
    rows = db.fetchall(
        """
        SELECT
            validation_run_id,
            model_name,
            artifact_path,
            hyperparameters_json,
            feature_spec_json,
            validation_metrics_json,
            test_metrics_json,
            split_config_json,
            is_selected,
            created_at
        FROM model_validation_runs
        WHERE model_name = ? AND is_selected = 1
        ORDER BY validation_run_id DESC
        LIMIT 1
        """,
        (model_name,),
    )
    if not rows:
        return None
    return ModelValidationRunRecord(**rows[0])


def get_best_selected_model_validation_run(
    db: Database,
    *,
    primary_metric: str = "f1",
) -> ModelValidationRunRecord:
    selected_records = [
        record for record in list_model_validation_runs(db) if record.is_selected
    ]
    if not selected_records:
        raise RuntimeError(
            "Inference requires at least one selected model. Run train mode first."
        )

    return max(
        selected_records,
        key=lambda record: float(
            json.loads(record.validation_metrics_json).get(primary_metric, float("-inf"))
        ),
    )
