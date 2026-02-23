CREATE TABLE IF NOT EXISTS raw_data (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL REFERENCES raw_batches(batch_id),
    row_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_data_batch ON raw_data(batch_id);
