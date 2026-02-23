CREATE TABLE IF NOT EXISTS raw_batches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id   TEXT    NOT NULL UNIQUE,
    n_rows     INTEGER NOT NULL,
    created_at TEXT    DEFAULT (datetime('now'))
);
