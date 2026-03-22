CREATE TABLE IF NOT EXISTS model_validation_runs (
    validation_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL CHECK (model_name IN ('catboost', 'mlp')),
    artifact_path TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    feature_spec_json TEXT NOT NULL,
    validation_metrics_json TEXT NOT NULL,
    test_metrics_json TEXT,
    split_config_json TEXT NOT NULL,
    is_selected INTEGER NOT NULL DEFAULT 0 CHECK (is_selected IN (0, 1)),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_model_validation_runs_model_name
ON model_validation_runs(model_name);

CREATE INDEX IF NOT EXISTS idx_model_validation_runs_selected
ON model_validation_runs(model_name, is_selected);
