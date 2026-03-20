import json
from collections.abc import Iterator

import pandas as pd

from src.database.connection import Database
from src.database.time_queries import get_row_ids_for_exact_dates

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


def load_split_dataframe(
    db: Database,
    split_row_ids: list[int],
    *,
    chunk_size: int = _SQLITE_IN_CLAUSE_CHUNK,
) -> pd.DataFrame:
    return load_rows_by_ids(db, split_row_ids, chunk_size=chunk_size)


def iter_ml_batches(
    db: Database,
    batch_dates: list[str],
    *,
    chunk_size: int = _SQLITE_IN_CLAUSE_CHUNK,
) -> Iterator[tuple[str, pd.DataFrame]]:
    for batch_date in batch_dates:
        row_ids = get_row_ids_for_exact_dates(db, [batch_date])
        yield batch_date, load_rows_by_ids(db, row_ids, chunk_size=chunk_size)
