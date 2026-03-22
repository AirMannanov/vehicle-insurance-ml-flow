# Model Validation Report: mlp

## Summary

- **Model version:** v2
- **Validation run id:** 2
- **Model family:** mlp
- **Selected:** no
- **Created at:** 2026-03-22 20:29:12
- **Artifact path:** `artifacts/models/mlp/20260322_232912.joblib`

## Metrics

| Split | Accuracy | Precision | Recall | F1 | N samples | Latency ms |
|-------|----------|-----------|--------|----|-----------|------------|
| validation | 0.7790 | 0.1957 | 0.6020 | 0.2954 | 140518 | 52.9396 |
| test | 0.7825 | 0.1485 | 0.5664 | 0.2353 | 147238 | 69.3503 |

## Chart

![Validation vs test metrics](../figures/models/model_report_mlp_v2.png)

## Class Balance

| Split | True positives | Pred positives | Positive rate | Pred positive rate | Zero baseline accuracy |
|-------|----------------|----------------|---------------|--------------------|------------------------|
| validation | 10811 | 33254 | 0.0769 | 0.2367 | 0.9231 |
| test | 8699 | 33185 | 0.0591 | 0.2254 | 0.9409 |

## Confusion Matrix

| Split | TP | TN | FP | FN |
|-------|----|----|----|----|
| validation | 6508 | 102961 | 26746 | 4303 |
| test | 4927 | 110281 | 28258 | 3772 |

## Split

- **Train ratio:** 0.7
- **Validation ratio:** 0.15
- **Test ratio:** 0.15
- **Train dates:** 1789
- **Validation dates:** 383
- **Test dates:** 384
- **Train rows:** 514280

## Feature Space

- **Feature columns:** 8
- **Numeric features:** 1
- **Categorical features:** 7
- **Dropped columns:** CARRYING_CAPACITY, CCM_TON, INSURED_VALUE, OBJECT_ID, PROD_YEAR, SEATS_NUM

## Hyperparameters

```json
{
  "activation": "relu",
  "alpha": 0.0001,
  "batch_size": 256,
  "early_stopping": true,
  "hidden_layer_sizes": [
    128,
    64
  ],
  "learning_rate_init": 0.001,
  "max_iter": 30,
  "random_state": 42,
  "solver": "adam"
}
```
