CREATE TABLE IF NOT EXISTS assoc_rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    TEXT    NOT NULL REFERENCES raw_batches(batch_id),
    antecedents TEXT    NOT NULL,
    consequents TEXT    NOT NULL,
    support     REAL    NOT NULL,
    confidence  REAL    NOT NULL,
    lift        REAL    NOT NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_assoc_rules_batch ON assoc_rules(batch_id);
