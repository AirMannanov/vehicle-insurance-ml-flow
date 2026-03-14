from typing import Literal

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
) -> tuple[list[str], list[str]]:
    """Split cleaned DataFrame columns into numeric and categorical feature lists.

    Excludes time_column and target_column. Uses dataset_schema for types.
    Only includes columns that exist in df (so safe after clean_batch dropped some).

    Returns:
        (numeric_features, categorical_features)
    """
    exclude = {time_column, target_column}
    feature_cols = [c for c in df.columns if c not in exclude]

    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for col in feature_cols:
        if not column_in_schema(col):
            continue
        if get_feature_type(col) == FeatureType.NUMERIC:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def _make_numeric_pipeline(scaler: Literal["standard", "minmax"]) -> Pipeline:
    scale_cls = StandardScaler if scaler == "standard" else MinMaxScaler
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scale_cls()),
    ])


def _make_categorical_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scaler: Literal["standard", "minmax"] = "standard",
) -> ColumnTransformer:
    """Build a ColumnTransformer for MLP-style preprocessing.

    - Numeric: impute with median, then scale (standard or minmax).
    - Categorical: impute with constant "missing", then one-hot encode.

    If either list is empty, that branch is omitted. remainder="drop" so only
    the specified columns are transformed.
    """
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        transformers.append(
            ("num", _make_numeric_pipeline(scaler), numeric_features),
        )
    if categorical_features:
        transformers.append(
            ("cat", _make_categorical_pipeline(), categorical_features),
        )

    if not transformers:
        raise ValueError(
            "At least one of numeric_features or categorical_features must be non-empty"
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
    )
