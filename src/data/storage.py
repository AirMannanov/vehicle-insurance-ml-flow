import json
import logging

import pandas as pd

from src.database.connection import Database
from src.data.time_utils import parse_event_time

logger = logging.getLogger("mlops")


def save_batch(
    db: Database,
    batch_id: str,
    df: pd.DataFrame,
    *,
    time_column: str | None = None,
) -> bool:
    """Save a single batch to the database. Returns True if inserted, False if already exists."""
    existing = db.fetchall(
        "SELECT 1 FROM raw_batches WHERE batch_id = ?", (batch_id,)
    )
    if existing:
        return False

    rows = []
    skipped_rows = 0
    for row in df.to_dict(orient="records"):
        event_time = None
        event_date = None
        if time_column is not None:
            event_time, event_date = parse_event_time(row, time_column)
            if event_time is None or event_date is None:
                skipped_rows += 1
                continue
        rows.append(
            (
                batch_id,
                json.dumps(row, default=str),
                event_time,
                event_date,
            )
        )

    if not rows:
        logger.warning(
            "Skipping batch %s: no rows with valid %s",
            batch_id,
            time_column or "event time",
        )
        return False

    db.execute(
        "INSERT INTO raw_batches (batch_id, n_rows) VALUES (?, ?)",
        (batch_id, len(rows)),
    )

    db.connection.executemany(
        """
        INSERT INTO raw_data (batch_id, row_json, event_time, event_date)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    db.commit()
    if skipped_rows:
        logger.warning(
            "Batch %s: skipped %d rows without valid %s",
            batch_id,
            skipped_rows,
            time_column or "event time",
        )
    return True


def save_all_batches(
    db: Database,
    batches: list[tuple[str, pd.DataFrame]],
    *,
    time_column: str | None = None,
) -> int:
    inserted = 0
    for batch_id, df in batches:
        if save_batch(db, batch_id, df, time_column=time_column):
            inserted += 1
        else:
            existing = db.fetchall(
                "SELECT 1 FROM raw_batches WHERE batch_id = ?",
                (batch_id,),
            )
            if existing:
                logger.warning("Batch %s already exists in raw_batches", batch_id)

    total = db.execute("SELECT COUNT(*) FROM raw_batches").fetchone()[0]
    logger.info("Saved %d new batches (total in DB: %d)", inserted, total)
    return inserted


def load_batch(db: Database, batch_id: str) -> pd.DataFrame:
    rows = db.fetchall(
        "SELECT row_json FROM raw_data WHERE batch_id = ?", (batch_id,)
    )
    if not rows:
        raise ValueError(f"Batch '{batch_id}' not found in raw_data")
    return pd.DataFrame([json.loads(r["row_json"]) for r in rows])


def list_batches(db: Database) -> list[dict]:
    return db.fetchall("SELECT * FROM raw_batches ORDER BY batch_id")
