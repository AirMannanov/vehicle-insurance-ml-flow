from dataclasses import dataclass, field
from typing import Literal

from src.database import (
    Database,
    get_row_ids_for_exact_dates,
    list_available_event_dates,
)


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    time_granularity: Literal["day"] = "day"
    min_unique_periods: int = 3

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Split ratios must sum to 1.0 "
                f"(got train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All split ratios must be positive")
        if self.time_granularity != "day":
            raise ValueError(
                f"Unsupported time_granularity: {self.time_granularity!r}; only 'day' is supported"
            )
        if self.min_unique_periods < 3:
            raise ValueError("min_unique_periods must be at least 3")


@dataclass(slots=True)
class SplitResult:
    train_dates: list[str]
    val_dates: list[str]
    test_dates: list[str]
    train_row_ids: list[int] = field(default_factory=list)
    val_row_ids: list[int] = field(default_factory=list)
    test_row_ids: list[int] = field(default_factory=list)


def _prepare_dates(dates: list[str]) -> list[str]:
    return sorted(set(dates))


def split_event_dates(
    dates: list[str],
    config: SplitConfig,
) -> SplitResult:
    unique_dates = _prepare_dates(dates)
    n_dates = len(unique_dates)
    if n_dates < config.min_unique_periods:
        raise ValueError(
            f"Need at least {config.min_unique_periods} unique dates for splitting; got {n_dates}"
        )

    train_end = max(1, min(n_dates - 2, int(n_dates * config.train_ratio)))
    val_end = max(
        train_end + 1,
        min(n_dates - 1, int(n_dates * (config.train_ratio + config.val_ratio))),
    )

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    if not train_dates or not val_dates or not test_dates:
        raise ValueError(
            "Split produced an empty partition; check ratios and the number of unique dates"
        )

    return SplitResult(
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
    )


def build_split_from_dates(
    db: Database,
    dates: list[str],
    config: SplitConfig | None = None,
) -> SplitResult:
    config = config or SplitConfig()
    split_result = split_event_dates(dates, config)
    split_result.train_row_ids = get_row_ids_for_exact_dates(db, split_result.train_dates)
    split_result.val_row_ids = get_row_ids_for_exact_dates(db, split_result.val_dates)
    split_result.test_row_ids = get_row_ids_for_exact_dates(db, split_result.test_dates)
    return split_result


def build_split_from_db(
    db: Database,
    config: SplitConfig | None = None,
) -> SplitResult:
    config = config or SplitConfig()
    split_result = split_event_dates(list_available_event_dates(db), config)

    split_result.train_row_ids = get_row_ids_for_exact_dates(db, split_result.train_dates)
    split_result.val_row_ids = get_row_ids_for_exact_dates(db, split_result.val_dates)
    split_result.test_row_ids = get_row_ids_for_exact_dates(db, split_result.test_dates)
    return split_result


def list_test_ml_batches(split_result: SplitResult) -> list[str]:
    return list(split_result.test_dates)
