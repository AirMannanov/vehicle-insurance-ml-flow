from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


def save_model_bundle(
    model_name: str,
    bundle: dict[str, Any],
    *,
    artifacts_dir: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(artifacts_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{timestamp}.joblib"
    joblib.dump(bundle, artifact_path)
    return str(artifact_path)


def load_model_bundle(artifact_path: str) -> dict[str, Any]:
    return joblib.load(artifact_path)
