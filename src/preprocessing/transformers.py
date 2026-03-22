from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.database.dataset_schema import FeatureType, column_in_schema, get_feature_type


def get_feature_columns(
    df: pd.DataFrame,
    *,
    time_column: str,
    target_column: str,
    numeric_override: list[str] | None = None,
    categorical_override: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split cleaned DataFrame columns into numeric and categorical feature lists."""
    exclude = {time_column, target_column}
    feature_cols = [c for c in df.columns if c not in exclude]

    if numeric_override is not None and categorical_override is not None:
        overlap = set(numeric_override) & set(categorical_override)
        if overlap:
            raise ValueError(
                f"Feature overrides overlap between numeric and categorical: {sorted(overlap)}"
            )

    numeric_features: list[str] = []
    categorical_features: list[str] = []

    if numeric_override is not None:
        numeric_features = [c for c in numeric_override if c in feature_cols]
    if categorical_override is not None:
        categorical_features = [c for c in categorical_override if c in feature_cols]

    skip_derived = set(numeric_features) | set(categorical_features)
    for col in feature_cols:
        if col in skip_derived:
            continue
        if not column_in_schema(col):
            continue
        feature_type = get_feature_type(col)
        if numeric_override is None and feature_type == FeatureType.NUMERIC:
            numeric_features.append(col)
        elif categorical_override is None and feature_type == FeatureType.CATEGORICAL:
            categorical_features.append(col)

    return numeric_features, categorical_features


def build_numeric_transformer(
    scaler: Literal["standard", "minmax"],
) -> Pipeline:
    scale_cls = StandardScaler if scaler == "standard" else MinMaxScaler
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scale_cls()),
    ])


def build_categorical_transformer() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scaler: Literal["standard", "minmax"] = "standard",
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            ("num", build_numeric_transformer(scaler), numeric_features),
        )
    if categorical_features:
        transformers.append(
            ("cat", build_categorical_transformer(), categorical_features),
        )

    if not transformers:
        raise ValueError(
            "At least one of numeric_features or categorical_features must be non-empty"
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )


def extract_preprocessor_feature_groups(
    preprocessor: ColumnTransformer,
) -> tuple[list[str], list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    transformers = getattr(preprocessor, "transformers_", preprocessor.transformers)
    for name, _transformer, columns in transformers:
        if name == "num":
            numeric_features = list(columns)
        elif name == "cat":
            categorical_features = list(columns)

    feature_columns = numeric_features + categorical_features
    return feature_columns, numeric_features, categorical_features


def ensure_feature_columns(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    missing_numeric = [column for column in numeric_features if column not in df.columns]
    missing_categorical = [
        column for column in categorical_features if column not in df.columns
    ]
    missing_columns = missing_numeric + missing_categorical
    if not missing_columns:
        return df

    out = df.copy()
    for column in missing_numeric:
        out[column] = np.nan
    for column in missing_categorical:
        out[column] = None
    return out
