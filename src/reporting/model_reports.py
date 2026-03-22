import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.database.model_validation_runs import ModelValidationRunRecord
from src.serving import load_model_bundle


def write_model_report(
    record: ModelValidationRunRecord,
    *,
    reports_dir: str = "reports/models",
    figures_dir: str = "reports/figures/models",
) -> str:
    bundle = load_model_bundle(record.artifact_path)
    validation_metrics = json.loads(record.validation_metrics_json)
    test_metrics = json.loads(record.test_metrics_json) if record.test_metrics_json else {}
    hyperparameters = json.loads(record.hyperparameters_json)
    feature_spec = json.loads(record.feature_spec_json)
    split_config = json.loads(record.split_config_json)
    training_metadata = bundle.get("training_metadata", {})

    report_dir = Path(reports_dir)
    figures_path = Path(figures_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    chart_path = figures_path / f"{record.model_name}_validation_run_{record.validation_run_id}.png"
    _write_metrics_chart(
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        output_path=chart_path,
    )

    report_path = report_dir / f"{record.model_name}_validation_run_{record.validation_run_id}.md"
    chart_rel_path = Path("..") / "figures" / "models" / chart_path.name
    report_content = _build_report_content(
        record=record,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        hyperparameters=hyperparameters,
        feature_spec=feature_spec,
        split_config=split_config,
        training_metadata=training_metadata,
        chart_rel_path=chart_rel_path.as_posix(),
    )
    report_path.write_text(report_content, encoding="utf-8")
    return str(report_path)


def _write_metrics_chart(
    *,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    val_values = [float(validation_metrics.get(metric, 0.0)) for metric in metric_names]
    test_values = [float(test_metrics.get(metric, 0.0)) for metric in metric_names]
    x = range(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([value - width / 2 for value in x], val_values, width=width, label="validation")
    ax.bar([value + width / 2 for value in x], test_values, width=width, label="test")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Validation vs test metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _build_report_content(
    *,
    record: ModelValidationRunRecord,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    feature_spec: dict[str, Any],
    split_config: dict[str, Any],
    training_metadata: dict[str, Any],
    chart_rel_path: str,
) -> str:
    lines = [
        f"# Model Validation Report: {record.model_name}",
        "",
        "## Summary",
        "",
        f"- **Validation run id:** {record.validation_run_id}",
        f"- **Model family:** {record.model_name}",
        f"- **Selected:** {'yes' if record.is_selected else 'no'}",
        f"- **Created at:** {record.created_at}",
        f"- **Artifact path:** `{record.artifact_path}`",
        "",
        "## Metrics",
        "",
        "| Split | Accuracy | Precision | Recall | F1 | N samples | Latency ms |",
        "|-------|----------|-----------|--------|----|-----------|------------|",
        _metric_row("validation", validation_metrics),
        _metric_row("test", test_metrics),
        "",
        "## Chart",
        "",
        f"![Validation vs test metrics]({chart_rel_path})",
        "",
        "## Class Balance",
        "",
        "| Split | True positives | Pred positives | Positive rate | Pred positive rate | Zero baseline accuracy |",
        "|-------|----------------|----------------|---------------|--------------------|------------------------|",
        _balance_row("validation", validation_metrics),
        _balance_row("test", test_metrics),
        "",
        "## Confusion Matrix",
        "",
        "| Split | TP | TN | FP | FN |",
        "|-------|----|----|----|----|",
        _confusion_row("validation", validation_metrics),
        _confusion_row("test", test_metrics),
        "",
        "## Split",
        "",
        f"- **Train ratio:** {split_config.get('train_ratio')}",
        f"- **Validation ratio:** {split_config.get('val_ratio')}",
        f"- **Test ratio:** {split_config.get('test_ratio')}",
        f"- **Train dates:** {len(training_metadata.get('train_dates', []))}",
        f"- **Validation dates:** {len(training_metadata.get('val_dates', []))}",
        f"- **Test dates:** {len(training_metadata.get('test_dates', []))}",
        f"- **Train rows:** {training_metadata.get('train_rows', 0)}",
        "",
        "## Feature Space",
        "",
        f"- **Feature columns:** {len(feature_spec.get('feature_columns', []))}",
        f"- **Numeric features:** {len(feature_spec.get('numeric_features', []))}",
        f"- **Categorical features:** {len(feature_spec.get('categorical_features', []))}",
        f"- **Dropped columns:** {', '.join(feature_spec.get('dropped_columns', [])) or 'none'}",
        "",
        "## Hyperparameters",
        "",
        "```json",
        json.dumps(hyperparameters, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _metric_row(split_name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {split_name} | "
        f"{float(metrics.get('accuracy', 0.0)):.4f} | "
        f"{float(metrics.get('precision', 0.0)):.4f} | "
        f"{float(metrics.get('recall', 0.0)):.4f} | "
        f"{float(metrics.get('f1', 0.0)):.4f} | "
        f"{int(metrics.get('n_samples', 0))} | "
        f"{float(metrics.get('inference_latency_ms', 0.0)):.4f} |"
    )


def _balance_row(split_name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {split_name} | "
        f"{int(metrics.get('true_positive_count', 0))} | "
        f"{int(metrics.get('pred_positive_count', 0))} | "
        f"{float(metrics.get('positive_rate', 0.0)):.4f} | "
        f"{float(metrics.get('pred_positive_rate', 0.0)):.4f} | "
        f"{float(metrics.get('baseline_accuracy_zero', 0.0)):.4f} |"
    )


def _confusion_row(split_name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {split_name} | "
        f"{int(metrics.get('tp', 0))} | "
        f"{int(metrics.get('tn', 0))} | "
        f"{int(metrics.get('fp', 0))} | "
        f"{int(metrics.get('fn', 0))} |"
    )
