CREATE TABLE IF NOT EXISTS model_registry (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL CHECK (model_name IN ('catboost', 'mlp')),
    hyperparameters_json TEXT NOT NULL,
    feature_spec_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_model_registry_model_name
ON model_registry(model_name);
