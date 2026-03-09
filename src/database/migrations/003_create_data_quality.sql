CREATE TABLE IF NOT EXISTS data_quality (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id     TEXT    NOT NULL REFERENCES raw_batches(batch_id),
    feature      TEXT    NOT NULL,
    missing_rate REAL    NOT NULL,
    unique_count INTEGER NOT NULL,
    value_type   TEXT    NOT NULL CHECK (value_type IN ('numeric', 'categorical')),
    stats_json   TEXT,
    UNIQUE(batch_id, feature)
);

CREATE INDEX IF NOT EXISTS idx_data_quality_batch ON data_quality(batch_id);
