"""Batch persistence: metadata → raw_batches, rows → raw_data."""

import json
import logging

import pandas as pd

from src.database.connection import Database

logger = logging.getLogger("mlops")


def save_batch(db: Database, batch_id: str, df: pd.DataFrame) -> bool:
    """Save a single batch to the database. Returns True if inserted, False if already exists."""
    existing = db.fetchall(
        "SELECT 1 FROM raw_batches WHERE batch_id = ?", (batch_id,)
    )
    if existing:
        return False

    db.execute(
        "INSERT INTO raw_batches (batch_id, n_rows) VALUES (?, ?)",
        (batch_id, len(df)),
    )

    rows = [
        (batch_id, json.dumps(row, default=str))
        for row in df.to_dict(orient="records")
    ]
    db.connection.executemany(
        "INSERT INTO raw_data (batch_id, row_json) VALUES (?, ?)",
        rows,
    )
    db.commit()
    return True


def save_all_batches(
    db: Database, batches: list[tuple[str, pd.DataFrame]]
) -> int:
    """Save a list of (batch_id, DataFrame) pairs. Returns count of newly inserted."""
    inserted = 0
    for batch_id, df in batches:
        if save_batch(db, batch_id, df):
            inserted += 1

    total = db.execute("SELECT COUNT(*) FROM raw_batches").fetchone()[0]
    logger.info("Saved %d new batches (total in DB: %d)", inserted, total)
    return inserted


def load_batch(db: Database, batch_id: str) -> pd.DataFrame:
    """Load a batch from raw_data back into a DataFrame."""
    rows = db.fetchall(
        "SELECT row_json FROM raw_data WHERE batch_id = ?", (batch_id,)
    )
    if not rows:
        raise ValueError(f"Batch '{batch_id}' not found in raw_data")
    return pd.DataFrame([json.loads(r["row_json"]) for r in rows])


def list_batches(db: Database) -> list[dict]:
    """Return all batch metadata records."""
    return db.fetchall("SELECT * FROM raw_batches ORDER BY batch_id")
