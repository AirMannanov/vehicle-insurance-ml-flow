from dataclasses import dataclass
from typing import Literal

from src.database import (
    Database,
    get_all_row_ids,
    get_row_ids_for_date_range,
    get_row_ids_for_exact_dates,
)
from src.tools import get_nested


@dataclass(slots=True)
class TrainDatasetConfig:
    selection_mode: Literal["all", "date_range", "exact_dates"] = "all"
    start_date: str | None = None
    end_date: str | None = None
    dates: list[str] | None = None


def get_train_dataset_config(train_config: dict) -> TrainDatasetConfig:
    dataset_config = get_nested(train_config, "dataset", default={}) or {}
    selection_mode = dataset_config.get("selection_mode", "all")
    if selection_mode not in {"all", "date_range", "exact_dates"}:
        raise ValueError(f"Unsupported dataset.selection_mode: {selection_mode!r}")
    dates = dataset_config.get("dates")
    if dates is not None and not isinstance(dates, list):
        raise ValueError("dataset.dates must be a list of YYYY-MM-DD strings")

    return TrainDatasetConfig(
        selection_mode=selection_mode,
        start_date=dataset_config.get("start_date"),
        end_date=dataset_config.get("end_date"),
        dates=dates,
    )


def select_training_row_ids(
    db: Database,
    train_config: dict,
) -> list[int]:
    dataset_config = get_train_dataset_config(train_config)
    if dataset_config.selection_mode == "all":
        return get_all_row_ids(db)

    if dataset_config.selection_mode == "date_range":
        if dataset_config.start_date is None and dataset_config.end_date is None:
            return get_all_row_ids(db)
        if dataset_config.start_date is None or dataset_config.end_date is None:
            raise ValueError(
                "dataset.start_date and dataset.end_date must both be set for date_range"
            )
        return get_row_ids_for_date_range(
            db,
            dataset_config.start_date,
            dataset_config.end_date,
        )
    if not dataset_config.dates:
        raise ValueError(
            "dataset.dates must be a non-empty list for selection_mode='exact_dates'"
        )
    return get_row_ids_for_exact_dates(db, dataset_config.dates)
