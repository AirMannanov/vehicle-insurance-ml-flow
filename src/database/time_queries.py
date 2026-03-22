from src.database.connection import Database


def get_all_row_ids(db: Database) -> list[int]:
    rows = db.fetchall(
        """
        SELECT id
        FROM raw_data
        ORDER BY event_time ASC, id ASC
        """
    )
    return [row["id"] for row in rows]


def list_available_event_dates(db: Database) -> list[str]:
    rows = db.fetchall(
        """
        SELECT DISTINCT event_date
        FROM raw_data
        WHERE event_date IS NOT NULL
        ORDER BY event_date ASC
        """
    )
    return [row["event_date"] for row in rows]


def get_available_event_dates_for_date_range(
    db: Database,
    start_date: str,
    end_date: str,
) -> list[str]:
    rows = db.fetchall(
        """
        SELECT DISTINCT event_date
        FROM raw_data
        WHERE event_date BETWEEN ? AND ?
        ORDER BY event_date ASC
        """,
        (start_date, end_date),
    )
    return [row["event_date"] for row in rows]


def get_available_event_dates_for_exact_dates(
    db: Database,
    dates: list[str],
) -> list[str]:
    if not dates:
        return []
    placeholders = ", ".join("?" for _ in dates)
    rows = db.fetchall(
        f"""
        SELECT DISTINCT event_date
        FROM raw_data
        WHERE event_date IN ({placeholders})
        ORDER BY event_date ASC
        """,
        dates,
    )
    return [row["event_date"] for row in rows]


def list_available_event_times(db: Database) -> list[str]:
    rows = db.fetchall(
        """
        SELECT DISTINCT event_time
        FROM raw_data
        WHERE event_time IS NOT NULL
        ORDER BY event_time ASC
        """
    )
    return [row["event_time"] for row in rows]


def get_row_ids_for_date_range(
    db: Database,
    start_date: str,
    end_date: str,
) -> list[int]:
    rows = db.fetchall(
        """
        SELECT id
        FROM raw_data
        WHERE event_date BETWEEN ? AND ?
        ORDER BY event_time ASC, id ASC
        """,
        (start_date, end_date),
    )
    return [row["id"] for row in rows]


def get_row_ids_for_exact_dates(
    db: Database,
    dates: list[str],
) -> list[int]:
    if not dates:
        return []

    placeholders = ", ".join("?" for _ in dates)
    rows = db.fetchall(
        f"""
        SELECT id
        FROM raw_data
        WHERE event_date IN ({placeholders})
        ORDER BY event_time ASC, id ASC
        """,
        dates,
    )
    return [row["id"] for row in rows]


def get_row_ids_for_source_batches(
    db: Database,
    batch_ids: list[str],
) -> list[int]:
    if not batch_ids:
        return []

    placeholders = ", ".join("?" for _ in batch_ids)
    rows = db.fetchall(
        f"""
        SELECT id
        FROM raw_data
        WHERE batch_id IN ({placeholders})
        ORDER BY event_time ASC, id ASC
        """,
        batch_ids,
    )
    return [row["id"] for row in rows]


def get_event_date_counts(db: Database) -> list[dict]:
    return db.fetchall(
        """
        SELECT event_date, COUNT(*) AS n_rows
        FROM raw_data
        WHERE event_date IS NOT NULL
        GROUP BY event_date
        ORDER BY event_date ASC
        """
    )
