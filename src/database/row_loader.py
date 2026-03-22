import json
from collections.abc import Iterator

import pandas as pd

from src.database.connection import Database

_SQLITE_IN_CLAUSE_CHUNK = 900


def _iter_chunks(values: list[int], chunk_size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def load_rows_by_ids(
    db: Database,
    row_ids: list[int],
    *,
    chunk_size: int = _SQLITE_IN_CLAUSE_CHUNK,
) -> pd.DataFrame:
    """Load raw_data rows for the given ids, preserving chronological order."""
    if not row_ids:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for chunk in _iter_chunks(row_ids, chunk_size):
        placeholders = ", ".join("?" for _ in chunk)
        rows = db.fetchall(
            f"""
            SELECT row_json
            FROM raw_data
            WHERE id IN ({placeholders})
            ORDER BY event_time ASC, id ASC
            """,
            chunk,
        )
        if not rows:
            continue
        frames.append(pd.DataFrame([json.loads(row["row_json"]) for row in rows]))

    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)
