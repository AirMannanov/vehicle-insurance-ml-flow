from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.cleaning import CleaningPlan
from src.database import Database, load_rows_by_ids
from src.preprocessing import prepare_features_for_catboost, prepare_features_for_mlp
from src.validation.metrics import compute_classification_metrics


@dataclass(slots=True)
class BatchEvaluation:
    batch_date: str
    metrics: dict[str, Any]


def select_best_threshold(
    model_name: str,
    *,
    model: Any,
    preprocessor: Any,
    df: pd.DataFrame,
    base_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    candidate_thresholds: list[float],
    primary_metric: str,
) -> tuple[float, dict[str, Any]]:
    prepared = _prepare_for_model(
        model_name,
        df=df,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        preprocessor=preprocessor,
    )
    probabilities, latency_ms = _predict_positive_probabilities(model, prepared.X)

    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in candidate_thresholds:
        y_pred = apply_threshold(probabilities, threshold)
        metrics = compute_classification_metrics(
            prepared.y,
            y_pred,
            inference_latency_ms=latency_ms,
        )
        metrics["threshold"] = threshold
        if best_metrics is None or _is_better(metrics, best_metrics, primary_metric):
            best_threshold = threshold
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("Failed to choose a threshold from candidate_thresholds")
    return best_threshold, best_metrics


def evaluate_model_dataframe(
    model_name: str,
    *,
    model: Any,
    preprocessor: Any,
    df: pd.DataFrame,
    base_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    threshold: float = 0.5,
) -> dict[str, Any]:
    prepared = _prepare_for_model(
        model_name,
        df=df,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        preprocessor=preprocessor,
    )
    probabilities, latency_ms = _predict_positive_probabilities(model, prepared.X)
    y_pred = apply_threshold(probabilities, threshold)
    metrics = compute_classification_metrics(
        prepared.y,
        y_pred,
        inference_latency_ms=latency_ms,
    )
    metrics["threshold"] = threshold
    return metrics


def evaluate_model_test_batches(
    db: Database,
    *,
    model_name: str,
    model: Any,
    preprocessor: Any,
    test_dates: list[str],
    base_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    threshold: float = 0.5,
) -> tuple[list[BatchEvaluation], dict[str, Any]]:
    batch_evaluations: list[BatchEvaluation] = []
    all_y_true: list[Any] = []
    all_y_pred: list[Any] = []
    total_latency_ms = 0.0

    for batch_date in test_dates:
        rows = db.fetchall(
            """
            SELECT id
            FROM raw_data
            WHERE event_date = ?
            ORDER BY event_time ASC, id ASC
            """,
            (batch_date,),
        )
        row_ids = [row["id"] for row in rows]
        if not row_ids:
            continue

        df = load_rows_by_ids(db, row_ids)
        prepared = _prepare_for_model(
            model_name,
            df=df,
            base_config=base_config,
            cleaning_plan=cleaning_plan,
            preprocessor=preprocessor,
        )
        probabilities, latency_ms = _predict_positive_probabilities(model, prepared.X)
        y_pred = apply_threshold(probabilities, threshold)
        metrics = compute_classification_metrics(
            prepared.y,
            y_pred,
            inference_latency_ms=latency_ms,
        )
        metrics["threshold"] = threshold
        batch_evaluations.append(BatchEvaluation(batch_date=batch_date, metrics=metrics))
        all_y_true.extend(prepared.y.tolist())
        all_y_pred.extend(list(y_pred))
        total_latency_ms += latency_ms

    aggregate_metrics = compute_classification_metrics(
        all_y_true,
        all_y_pred,
        inference_latency_ms=total_latency_ms,
    )
    aggregate_metrics["threshold"] = threshold
    return batch_evaluations, aggregate_metrics


def _prepare_for_model(
    model_name: str,
    *,
    df: pd.DataFrame,
    base_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
    preprocessor: Any,
):
    if model_name == "mlp":
        return prepare_features_for_mlp(
            df,
            base_config,
            cleaning_plan=cleaning_plan,
            fit=False,
            preprocessor=preprocessor,
        )
    if model_name == "catboost":
        return prepare_features_for_catboost(
            df,
            base_config,
            cleaning_plan=cleaning_plan,
        )
    raise ValueError(f"Unsupported model_name: {model_name!r}")


def _predict_positive_probabilities(model: Any, X) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        raw_predictions = np.asarray(model.predict(X))
        probabilities = raw_predictions.astype(float)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return np.asarray(probabilities, dtype=float), latency_ms


def apply_threshold(probabilities, threshold: float):
    probabilities_arr = np.asarray(probabilities, dtype=float)
    return (probabilities_arr >= threshold).astype(int)


def _is_better(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    primary_metric: str,
) -> bool:
    candidate_score = float(candidate_metrics[primary_metric])
    current_score = float(current_metrics[primary_metric])
    if candidate_score != current_score:
        return candidate_score > current_score

    candidate_recall = float(candidate_metrics.get("recall", 0.0))
    current_recall = float(current_metrics.get("recall", 0.0))
    if candidate_recall != current_recall:
        return candidate_recall > current_recall

    candidate_pred_positive = int(candidate_metrics.get("pred_positive_count", 0))
    current_pred_positive = int(current_metrics.get("pred_positive_count", 0))
    return candidate_pred_positive > current_pred_positive
