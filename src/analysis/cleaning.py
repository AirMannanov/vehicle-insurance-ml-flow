import logging

import pandas as pd

from src.analysis.data_quality import DQRow

logger = logging.getLogger("mlops")


def clean_batch(
    df: pd.DataFrame,
    dq_rows: list[DQRow],
    *,
    max_missing_rate: float,
    min_unique_ratio: float | None,
    max_unique_ratio: float | None,
) -> pd.DataFrame:
    """Drop columns that fail quality thresholds.

    Uses precomputed DQ metrics. Columns are dropped if:
    - missing_rate > max_missing_rate, or
    - unique_count / n_rows < min_unique_ratio (constants / near-constants), or
    - unique_count / n_rows > max_unique_ratio (near-keys).

    Returns DataFrame with only columns that pass all thresholds. If min/max_unique_ratio
    is None, that check is skipped.
    """
    n_rows = len(df)
    if n_rows == 0:
        return df.copy()

    dq_by_feature = {r.feature: r for r in dq_rows}
    to_drop = []

    for col in df.columns:
        if col not in dq_by_feature:
            continue
        row = dq_by_feature[col]
        if row.missing_rate > max_missing_rate:
            to_drop.append(col)
            logger.debug("Dropping %s: missing_rate %.2f > %.2f", col, row.missing_rate, max_missing_rate)
            continue
        unique_ratio = row.unique_count / n_rows
        if min_unique_ratio is not None and unique_ratio < min_unique_ratio:
            to_drop.append(col)
            logger.debug("Dropping %s: unique_ratio %.4f < %.4f", col, unique_ratio, min_unique_ratio)
            continue
        if max_unique_ratio is not None and unique_ratio > max_unique_ratio:
            to_drop.append(col)
            logger.debug("Dropping %s: unique_ratio %.4f > %.4f", col, unique_ratio, max_unique_ratio)

    keep = [c for c in df.columns if c not in to_drop]
    if to_drop:
        logger.info("Cleaning: dropped %d columns by thresholds: %s", len(to_drop), to_drop)
    return df[keep].copy()
