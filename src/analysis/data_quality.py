import json
import logging
from dataclasses import dataclass

import pandas as pd

from src.database.connection import Database
from src.database.dataset_schema import ValueType, get_value_type

logger = logging.getLogger("mlops")


@dataclass
class DQRow:
    """Data quality metrics for a single feature in a batch."""
    feature: str
    missing_rate: float
    unique_count: int
    value_type: ValueType
    stats_json: str | None


def compute_batch_dq(df: pd.DataFrame) -> list[DQRow]:
    """Compute data quality metrics for each column of the batch.

    Returns a list of DQRow. stats_json: for numeric — min, max, mean, std;
    for categorical — top 10 value counts (JSON string). Uses dataset schema (get_value_type).
    """
    rows = []
    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean())
        unique_count = int(s.nunique())
        value_type: ValueType = get_value_type(col)

        if value_type == "numeric":
            valid = s.dropna()
            if len(valid) == 0:
                stats = {}
            else:
                stats = {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()) if len(valid) > 1 else 0.0,
                }
        else:
            top = s.value_counts().head(10)
            stats = {str(k): int(v) for k, v in top.items()}

        stats_str = json.dumps(stats, default=str) if stats else None
        rows.append(
            DQRow(
                feature=col,
                missing_rate=missing_rate,
                unique_count=unique_count,
                value_type=value_type,
                stats_json=stats_str,
            )
        )
    return rows


def save_batch_dq(db: Database, batch_id: str, dq_rows: list[DQRow]) -> None:
    """Write data quality rows for one batch into data_quality table.

    Replaces any existing rows for this batch_id.
    """
    db.execute("DELETE FROM data_quality WHERE batch_id = ?", (batch_id,))
    for row in dq_rows:
        db.execute(
            """INSERT INTO data_quality (batch_id, feature, missing_rate, unique_count, value_type, stats_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (batch_id, row.feature, row.missing_rate, row.unique_count, row.value_type, row.stats_json),
        )
    db.commit()
    logger.debug("Saved DQ for batch %s: %d features", batch_id, len(dq_rows))
