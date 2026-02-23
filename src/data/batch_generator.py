"""Date-based batch splitting."""

import logging

import pandas as pd

logger = logging.getLogger("mlops")


def generate_batches(
    df: pd.DataFrame,
    time_column: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Split *df* into monthly batches based on *time_column*.

    Returns a list of ``(batch_id, batch_df)`` tuples sorted by period.
    """
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], dayfirst=False)

    batches: list[tuple[str, pd.DataFrame]] = []
    for period, group in df.groupby(df[time_column].dt.to_period("M")):
        batch_id = str(period)
        batches.append((batch_id, group))
        logger.info("Batch %s: %d rows", batch_id, len(group))

    logger.info("Generated %d batches", len(batches))
    return batches
