from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

from src.analysis.cleaning import CleaningPlan, build_cleaning_plan
from src.database import Database, insert_model_registry_record, load_rows_by_ids
from src.models.training_dataset import select_training_row_ids
from src.preprocessing import prepare_features_for_catboost, prepare_features_for_mlp
from src.serving import save_model_bundle
from src.tools import get_nested

logger = logging.getLogger("mlops")


@dataclass(slots=True)
class TrainResult:
    model_name: str
    model_id: int
    artifact_path: str
    n_rows: int


def train_models(
    db: Database,
    base_config: dict[str, Any],
    train_config: dict[str, Any],
) -> list[TrainResult]:
    row_ids = select_training_row_ids(db, train_config)
    if not row_ids:
        raise ValueError("No rows matched the train dataset selection")

    df = load_rows_by_ids(db, row_ids)
    if df.empty:
        raise ValueError("Selected train dataset is empty")

    cleaning_plan = _build_training_cleaning_plan(df, base_config)
    enabled_models = get_nested(train_config, "models", "enabled", default=[]) or []
    if not enabled_models:
        raise ValueError("train_config.models.enabled must contain at least one model")

    artifacts_dir = get_nested(
        train_config,
        "output",
        "artifacts_dir",
        default="artifacts/models",
    )
    results: list[TrainResult] = []

    for model_name in enabled_models:
        if model_name == "mlp":
            results.append(
                _train_mlp(
                    db,
                    df,
                    row_ids,
                    base_config,
                    train_config,
                    cleaning_plan,
                    artifacts_dir=artifacts_dir,
                )
            )
            continue
        if model_name == "catboost":
            results.append(
                _train_catboost(
                    db,
                    df,
                    row_ids,
                    base_config,
                    train_config,
                    cleaning_plan,
                    artifacts_dir=artifacts_dir,
                )
            )
            continue
        raise ValueError(f"Unsupported model in train_config.models.enabled: {model_name!r}")

    return results


def _build_training_cleaning_plan(
    df,
    base_config: dict[str, Any],
) -> CleaningPlan:
    time_column = get_nested(base_config, "data", "time_column", default="INSR_BEGIN")
    target_column = get_nested(base_config, "data", "target_column", default="CLAIM_PAID")
    feature_df = df.drop(columns=[time_column, target_column], errors="ignore")
    return build_cleaning_plan(
        feature_df,
        max_missing_rate=get_nested(
            base_config,
            "cleaning",
            "max_missing_rate",
            default=0.5,
        ),
        min_unique_ratio=get_nested(
            base_config,
            "cleaning",
            "min_unique_ratio",
            default=0.10,
        ),
        max_unique_ratio=get_nested(
            base_config,
            "cleaning",
            "max_unique_ratio",
            default=0.90,
        ),
    )


def _train_mlp(
    db: Database,
    df,
    row_ids: list[int],
    base_config: dict[str, Any],
    train_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    *,
    artifacts_dir: str,
) -> TrainResult:
    hyperparameters = (get_nested(train_config, "mlp", default={}) or {}).copy()
    prepared = prepare_features_for_mlp(
        df,
        base_config,
        cleaning_plan=cleaning_plan,
        fit=True,
    )
    model = MLPClassifier(**hyperparameters)
    model.fit(prepared.X, prepared.y)

    feature_spec = _build_feature_spec(
        base_config,
        prepared.feature_columns,
        prepared.numeric_features,
        prepared.categorical_features,
        prepared.dropped_columns,
    )
    bundle = _build_artifact_bundle(
        model_name="mlp",
        model=model,
        preprocessor=prepared.preprocessor,
        feature_spec=feature_spec,
        train_row_ids=row_ids,
        hyperparameters=hyperparameters,
        n_rows=len(df),
    )
    artifact_path = save_model_bundle("mlp", bundle, artifacts_dir=artifacts_dir)
    model_id = insert_model_registry_record(
        db,
        model_name="mlp",
        hyperparameters=hyperparameters,
        feature_spec=feature_spec,
        artifact_path=artifact_path,
    )
    logger.info("Trained mlp model_id=%s rows=%s artifact=%s", model_id, len(df), artifact_path)
    return TrainResult("mlp", model_id, artifact_path, len(df))


def _train_catboost(
    db: Database,
    df,
    row_ids: list[int],
    base_config: dict[str, Any],
    train_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    *,
    artifacts_dir: str,
) -> TrainResult:
    hyperparameters = (get_nested(train_config, "catboost", default={}) or {}).copy()
    hyperparameters.setdefault("allow_writing_files", False)
    prepared = prepare_features_for_catboost(
        df,
        base_config,
        cleaning_plan=cleaning_plan,
    )
    model = CatBoostClassifier(**hyperparameters)
    model.fit(
        prepared.X,
        prepared.y,
        cat_features=prepared.categorical_features,
    )

    feature_spec = _build_feature_spec(
        base_config,
        prepared.feature_columns,
        prepared.numeric_features,
        prepared.categorical_features,
        prepared.dropped_columns,
    )
    bundle = _build_artifact_bundle(
        model_name="catboost",
        model=model,
        preprocessor=None,
        feature_spec=feature_spec,
        train_row_ids=row_ids,
        hyperparameters=hyperparameters,
        n_rows=len(df),
    )
    artifact_path = save_model_bundle("catboost", bundle, artifacts_dir=artifacts_dir)
    model_id = insert_model_registry_record(
        db,
        model_name="catboost",
        hyperparameters=hyperparameters,
        feature_spec=feature_spec,
        artifact_path=artifact_path,
    )
    logger.info(
        "Trained catboost model_id=%s rows=%s artifact=%s",
        model_id,
        len(df),
        artifact_path,
    )
    return TrainResult("catboost", model_id, artifact_path, len(df))


def _build_feature_spec(
    base_config: dict[str, Any],
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    dropped_columns: list[str],
) -> dict[str, Any]:
    return {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": dropped_columns,
        "time_column": get_nested(base_config, "data", "time_column", default="INSR_BEGIN"),
        "target_column": get_nested(base_config, "data", "target_column", default="CLAIM_PAID"),
    }


def _build_artifact_bundle(
    *,
    model_name: str,
    model: Any,
    preprocessor: Any,
    feature_spec: dict[str, Any],
    train_row_ids: list[int],
    hyperparameters: dict[str, Any],
    n_rows: int,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model": model,
        "preprocessor": preprocessor,
        "feature_spec": feature_spec,
        "train_row_ids": train_row_ids,
        "hyperparameters": hyperparameters,
        "training_metadata": {
            "n_rows": n_rows,
        },
    }
