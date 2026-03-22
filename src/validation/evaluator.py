from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analysis.cleaning import CleaningPlan
from src.database import Database, load_rows_by_ids
from src.preprocessing import prepare_features_for_catboost, prepare_features_for_mlp
from src.validation.metrics import compute_classification_metrics


@dataclass(slots=True)
class BatchEvaluation:
    batch_date: str
    metrics: dict[str, Any]


def evaluate_model_dataframe(
    model_name: str,
    *,
    model: Any,
    preprocessor: Any,
    df: pd.DataFrame,
    base_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
) -> dict[str, Any]:
    prepared = _prepare_for_model(
        model_name,
        df=df,
        base_config=base_config,
        cleaning_plan=cleaning_plan,
        preprocessor=preprocessor,
    )
    started = time.perf_counter()
    y_pred = model.predict(prepared.X)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return compute_classification_metrics(
        prepared.y,
        y_pred,
        inference_latency_ms=latency_ms,
    )


def evaluate_model_test_batches(
    db: Database,
    *,
    model_name: str,
    model: Any,
    preprocessor: Any,
    test_dates: list[str],
    base_config: dict[str, Any],
    cleaning_plan: CleaningPlan,
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
        started = time.perf_counter()
        y_pred = model.predict(prepared.X)
        latency_ms = (time.perf_counter() - started) * 1000.0
        metrics = compute_classification_metrics(
            prepared.y,
            y_pred,
            inference_latency_ms=latency_ms,
        )
        batch_evaluations.append(BatchEvaluation(batch_date=batch_date, metrics=metrics))
        all_y_true.extend(prepared.y.tolist())
        all_y_pred.extend(list(y_pred))
        total_latency_ms += latency_ms

    aggregate_metrics = compute_classification_metrics(
        all_y_true,
        all_y_pred,
        inference_latency_ms=total_latency_ms,
    )
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
