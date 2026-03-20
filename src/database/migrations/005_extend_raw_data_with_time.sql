ALTER TABLE raw_data ADD COLUMN event_time TEXT;
ALTER TABLE raw_data ADD COLUMN event_date TEXT;

-- INSR_BEGIN is the dataset event-time column configured in config.yaml.
UPDATE raw_data
SET
    event_time = datetime(json_extract(row_json, '$.INSR_BEGIN')),
    event_date = date(json_extract(row_json, '$.INSR_BEGIN'));

DELETE FROM raw_data
WHERE event_time IS NULL OR event_date IS NULL;

UPDATE raw_batches
SET n_rows = (
    SELECT COUNT(*)
    FROM raw_data
    WHERE raw_data.batch_id = raw_batches.batch_id
);

DELETE FROM data_quality
WHERE batch_id IN (
    SELECT batch_id
    FROM raw_batches
    WHERE n_rows = 0
);

DELETE FROM assoc_rules
WHERE batch_id IN (
    SELECT batch_id
    FROM raw_batches
    WHERE n_rows = 0
);

DELETE FROM raw_batches
WHERE n_rows = 0;

CREATE INDEX IF NOT EXISTS idx_raw_data_event_date_id ON raw_data(event_date, id);
