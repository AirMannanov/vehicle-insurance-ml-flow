import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

logger = logging.getLogger("mlops")


def parse_event_time(
    raw_row: Mapping[str, Any],
    time_column: str,
) -> tuple[str | None, str | None]:
    """Extract normalized event timestamp and date from a raw row.

    Returns strings suitable for SQLite TEXT columns:
    - event_time: YYYY-MM-DD HH:MM:SS
    - event_date: YYYY-MM-DD
    """
    raw_value = raw_row.get(time_column)
    if raw_value is None or raw_value == "":
        return None, None

    parsed = pd.to_datetime(raw_value, errors="coerce", dayfirst=False, format="mixed")
    if pd.isna(parsed):
        parsed = pd.to_datetime(raw_value, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        logger.warning(
            "Failed to parse event time from column %s: %r",
            time_column,
            raw_value,
        )
        return None, None

    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)

    return (
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        timestamp.strftime("%Y-%m-%d"),
    )
