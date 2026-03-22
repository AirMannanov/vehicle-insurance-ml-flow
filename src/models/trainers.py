from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

from src.analysis.cleaning import CleaningPlan, build_cleaning_plan
from src.database import (
    Database,
    clear_selected_model_for_name,
    get_selected_model_validation_run,
    insert_model_validation_run,
    load_rows_by_ids,
    mark_model_validation_run_selected,
    update_model_validation_run_test_metrics,
)
from src.models.training_dataset import select_training_dates
from src.preprocessing import prepare_features_for_catboost, prepare_features_for_mlp
from src.serving import save_model_bundle
from src.tools import get_nested
from src.validation import (
    SplitConfig,
    build_split_from_dates,
    evaluate_model_test_batches,
    select_best_threshold,
)

logger = logging.getLogger("mlops")


@dataclass(slots=True)
class TrainResult:
    model_name: str
    validation_run_id: int
    artifact_path: str
    train_rows: int
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    is_selected: bool


def train_models(
    db: Database,
    base_config: dict[str, Any],
    train_config: dict[str, Any],
) -> list[TrainResult]:
    if not get_nested(train_config, "validation", "enabled", default=True):
        raise ValueError("train_config.validation.enabled must be true for train mode")

    selected_dates = select_training_dates(db, train_config)
    if not selected_dates:
        raise ValueError("No dates matched the train dataset selection")

    split_config = _get_split_config(train_config)
    split_result = build_split_from_dates(db, selected_dates, split_config)

    train_df = load_rows_by_ids(db, split_result.train_row_ids)
    val_df = load_rows_by_ids(db, split_result.val_row_ids)
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split must be non-empty")

    cleaning_plan = _build_training_cleaning_plan(train_df, base_config)
    enabled_models = get_nested(train_config, "models", "enabled", default=[]) or []
    if not enabled_models:
        raise ValueError("train_config.models.enabled must contain at least one model")

    artifacts_dir = get_nested(
        train_config,
        "output",
        "artifacts_dir",
        default="artifacts/models",
    )
    primary_metric = get_nested(train_config, "validation", "primary_metric", default="f1")
    candidate_thresholds = _get_candidate_thresholds(train_config)

    results: list[TrainResult] = []
    for model_name in enabled_models:
        if model_name == "mlp":
            results.append(
                _train_evaluate_select_mlp(
                    db,
                    train_df=train_df,
                    val_df=val_df,
                    split_result=split_result,
                    base_config=base_config,
                    train_config=train_config,
                    cleaning_plan=cleaning_plan,
                    split_config=split_config,
                    artifacts_dir=artifacts_dir,
                    primary_metric=primary_metric,
                    candidate_thresholds=candidate_thresholds,
                )
            )
            continue

        if model_name == "catboost":
            results.append(
                _train_evaluate_select_catboost(
                    db,
                    train_df=train_df,
                    val_df=val_df,
                    split_result=split_result,
                    base_config=base_config,
                    train_config=train_config,
                    cleaning_plan=cleaning_plan,
                    split_config=split_config,
                    artifacts_dir=artifacts_dir,
                    primary_metric=primary_metric,
                    candidate_thresholds=candidate_thresholds,
                )
            )
            continue

        raise ValueError(f"Unsupported model in train_config.models.enabled: {model_name!r}")

    return results


def _train_evaluate_select_mlp(
    db: Database,
    *,
    train_df,
    val_df,
    split_result,
    base_config: dict[str, Any],
    train_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    split_config: SplitConfig,
    artifacts_dir: str,
    primary_metric: str,
    candidate_thresholds: list[float],
) -> TrainResult:
    hyperparameters = (get_nested(train_config, "mlp", default={}) or {}).copy()
    prepared = prepare_features_for_mlp(
        train_df,
        base_config,
        cleaning_plan=cleaning_plan,
        fit=True,
    )
    model = MLPClassifier(**hyperparameters)
    model.fit(prepared.X, prepared.y)

    best_threshold, validation_metrics = select_best_threshold(
        "mlp",
        model=model,
        preprocessor=prepared.preprocessor,
        df=val_df,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        candidate_thresholds=candidate_thresholds,
        primary_metric=primary_metric,
    )
    _batch_results, test_metrics = evaluate_model_test_batches(
        db,
        model_name="mlp",
        model=model,
        preprocessor=prepared.preprocessor,
        test_dates=split_result.test_dates,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        threshold=best_threshold,
    )
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
        split_result=split_result,
        hyperparameters=hyperparameters,
        train_rows=len(train_df),
        threshold=best_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    artifact_path = save_model_bundle("mlp", bundle, artifacts_dir=artifacts_dir)
    validation_run_id = insert_model_validation_run(
        db,
        model_name="mlp",
        artifact_path=artifact_path,
        hyperparameters=hyperparameters,
        feature_spec=feature_spec,
        validation_metrics=validation_metrics,
        split_config=_serialize_split_config(split_config, threshold=best_threshold),
    )
    update_model_validation_run_test_metrics(
        db,
        validation_run_id=validation_run_id,
        test_metrics=test_metrics,
    )
    is_selected = _update_selected_model(
        db,
        model_name="mlp",
        validation_run_id=validation_run_id,
        validation_metrics=validation_metrics,
        primary_metric=primary_metric,
    )
    logger.info(
        "Trained mlp validation_run_id=%s train_rows=%s artifact=%s",
        validation_run_id,
        len(train_df),
        artifact_path,
    )
    return TrainResult(
        model_name="mlp",
        validation_run_id=validation_run_id,
        artifact_path=artifact_path,
        train_rows=len(train_df),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        is_selected=is_selected,
    )


def _train_evaluate_select_catboost(
    db: Database,
    *,
    train_df,
    val_df,
    split_result,
    base_config: dict[str, Any],
    train_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    split_config: SplitConfig,
    artifacts_dir: str,
    primary_metric: str,
    candidate_thresholds: list[float],
) -> TrainResult:
    hyperparameters = (get_nested(train_config, "catboost", default={}) or {}).copy()
    hyperparameters.setdefault("allow_writing_files", False)
    prepared = prepare_features_for_catboost(
        train_df,
        base_config,
        cleaning_plan=cleaning_plan,
    )
    model = CatBoostClassifier(**hyperparameters)
    model.fit(
        prepared.X,
        prepared.y,
        cat_features=prepared.categorical_features,
    )

    best_threshold, validation_metrics = select_best_threshold(
        "catboost",
        model=model,
        preprocessor=None,
        df=val_df,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        candidate_thresholds=candidate_thresholds,
        primary_metric=primary_metric,
    )
    _batch_results, test_metrics = evaluate_model_test_batches(
        db,
        model_name="catboost",
        model=model,
        preprocessor=None,
        test_dates=split_result.test_dates,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        threshold=best_threshold,
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
        split_result=split_result,
        hyperparameters=hyperparameters,
        train_rows=len(train_df),
        threshold=best_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )
    artifact_path = save_model_bundle("catboost", bundle, artifacts_dir=artifacts_dir)
    validation_run_id = insert_model_validation_run(
        db,
        model_name="catboost",
        artifact_path=artifact_path,
        hyperparameters=hyperparameters,
        feature_spec=feature_spec,
        validation_metrics=validation_metrics,
        split_config=_serialize_split_config(split_config, threshold=best_threshold),
    )
    update_model_validation_run_test_metrics(
        db,
        validation_run_id=validation_run_id,
        test_metrics=test_metrics,
    )
    is_selected = _update_selected_model(
        db,
        model_name="catboost",
        validation_run_id=validation_run_id,
        validation_metrics=validation_metrics,
        primary_metric=primary_metric,
    )
    logger.info(
        "Trained catboost validation_run_id=%s train_rows=%s artifact=%s",
        validation_run_id,
        len(train_df),
        artifact_path,
    )
    return TrainResult(
        model_name="catboost",
        validation_run_id=validation_run_id,
        artifact_path=artifact_path,
        train_rows=len(train_df),
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        is_selected=is_selected,
    )


def _build_training_cleaning_plan(
    df,
    base_config: dict[str, Any],
) -> CleaningPlan:
    time_column = get_nested(base_config, "data", "time_column", default="INSR_BEGIN")
    target_column = get_nested(base_config, "data", "target_column", default="CLAIM_PAID")
    feature_df = df.drop(columns=[time_column, target_column], errors="ignore")
    cleaning_plan = build_cleaning_plan(
        feature_df,
        max_missing_rate=get_nested(base_config, "cleaning", "max_missing_rate", default=0.5),
        min_unique_ratio=get_nested(base_config, "cleaning", "min_unique_ratio", default=0.10),
        max_unique_ratio=get_nested(base_config, "cleaning", "max_unique_ratio", default=0.90),
    )
    forced_drop_columns = get_nested(
        base_config,
        "cleaning",
        "forced_drop_columns",
        default=["OBJECT_ID"],
    ) or []
    dropped_columns = sorted(set(cleaning_plan.dropped_columns) | set(forced_drop_columns))
    return CleaningPlan(dropped_columns=dropped_columns)


def _get_split_config(train_config: dict[str, Any]) -> SplitConfig:
    return SplitConfig(
        train_ratio=float(get_nested(train_config, "validation", "train_ratio", default=0.70)),
        val_ratio=float(get_nested(train_config, "validation", "val_ratio", default=0.15)),
        test_ratio=float(get_nested(train_config, "validation", "test_ratio", default=0.15)),
    )


def _serialize_split_config(
    split_config: SplitConfig,
    *,
    threshold: float,
) -> dict[str, Any]:
    return {
        "train_ratio": split_config.train_ratio,
        "val_ratio": split_config.val_ratio,
        "test_ratio": split_config.test_ratio,
        "time_granularity": split_config.time_granularity,
        "min_unique_periods": split_config.min_unique_periods,
        "threshold": threshold,
    }


def _get_candidate_thresholds(train_config: dict[str, Any]) -> list[float]:
    thresholds = get_nested(train_config, "validation", "candidate_thresholds")
    if thresholds:
        return [float(value) for value in thresholds]
    return [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]


def _update_selected_model(
    db: Database,
    *,
    model_name: str,
    validation_run_id: int,
    validation_metrics: dict[str, Any],
    primary_metric: str,
) -> bool:
    new_score = float(validation_metrics[primary_metric])
    current_selected = get_selected_model_validation_run(db, model_name=model_name)
    if current_selected is not None:
        current_metrics = json.loads(current_selected.validation_metrics_json)
        current_score = float(current_metrics[primary_metric])
        if new_score <= current_score:
            return False

    clear_selected_model_for_name(db, model_name=model_name)
    mark_model_validation_run_selected(db, validation_run_id=validation_run_id)
    return True


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
    split_result,
    hyperparameters: dict[str, Any],
    train_rows: int,
    threshold: float,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model": model,
        "preprocessor": preprocessor,
        "feature_spec": feature_spec,
        "hyperparameters": hyperparameters,
        "training_metadata": {
            "train_rows": train_rows,
            "train_dates": list(split_result.train_dates),
            "val_dates": list(split_result.val_dates),
            "test_dates": list(split_result.test_dates),
            "threshold": threshold,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
