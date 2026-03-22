from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


def compute_classification_metrics(
    y_true,
    y_pred,
    *,
    inference_latency_ms: float | None = None,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    n_samples = int(len(y_true_arr))
    true_positive_count = int((y_true_arr == 1).sum())
    pred_positive_count = int((y_pred_arr == 1).sum())
    baseline_accuracy_zero = float((y_true_arr == 0).mean()) if n_samples else 0.0
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "n_samples": n_samples,
        "true_positive_count": true_positive_count,
        "pred_positive_count": pred_positive_count,
        "positive_rate": float(true_positive_count / n_samples) if n_samples else 0.0,
        "pred_positive_rate": float(pred_positive_count / n_samples) if n_samples else 0.0,
        "baseline_accuracy_zero": baseline_accuracy_zero,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "inference_latency_ms": inference_latency_ms,
    }
