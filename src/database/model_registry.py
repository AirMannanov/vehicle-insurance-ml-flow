import json
from dataclasses import dataclass
from typing import Any

from src.database.connection import Database


@dataclass(slots=True)
class ModelRegistryRecord:
    model_id: int
    model_name: str
    hyperparameters_json: str
    feature_spec_json: str
    artifact_path: str
    created_at: str


def insert_model_registry_record(
    db: Database,
    *,
    model_name: str,
    hyperparameters: dict[str, Any],
    feature_spec: dict[str, Any],
    artifact_path: str,
) -> int:
    cursor = db.execute(
        """
        INSERT INTO model_registry (
            model_name,
            hyperparameters_json,
            feature_spec_json,
            artifact_path
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            model_name,
            json.dumps(hyperparameters, sort_keys=True),
            json.dumps(feature_spec, sort_keys=True),
            artifact_path,
        ),
    )
    db.commit()
    return int(cursor.lastrowid)


def list_model_registry_records(
    db: Database,
) -> list[ModelRegistryRecord]:
    rows = db.fetchall(
        """
        SELECT
            model_id,
            model_name,
            hyperparameters_json,
            feature_spec_json,
            artifact_path,
            created_at
        FROM model_registry
        ORDER BY model_id ASC
        """
    )
    return [ModelRegistryRecord(**row) for row in rows]


def list_model_registry_records_by_name(
    db: Database,
    *,
    model_name: str,
) -> list[ModelRegistryRecord]:
    rows = db.fetchall(
        """
        SELECT
            model_id,
            model_name,
            hyperparameters_json,
            feature_spec_json,
            artifact_path,
            created_at
        FROM model_registry
        WHERE model_name = ?
        ORDER BY model_id ASC
        """,
        (model_name,),
    )
    return [ModelRegistryRecord(**row) for row in rows]
