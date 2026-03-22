from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from src.analysis.cleaning import clean_batch
from src.analysis.data_quality import DQRow
from src.preprocessing.transformers import get_feature_columns
from src.tools import get_nested


def extract_target(df: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in DataFrame")
    target = pd.to_numeric(df[target_column], errors="coerce")
    return target.fillna(0).gt(0).astype(int)


class Preprocessor(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.time_column = get_nested(config, "data", "time_column", default="INSR_BEGIN")
        self.target_column = get_nested(
            config, "data", "target_column", default="CLAIM_PAID"
        )

    def apply_optional_cleaning(
        self,
        df: pd.DataFrame,
        *,
        dq_rows: list[DQRow] | None,
    ) -> pd.DataFrame:
        if dq_rows is None:
            return df.copy()

        protected_columns = [
            c for c in (self.time_column, self.target_column) if c in df.columns
        ]
        feature_df = df.drop(columns=protected_columns, errors="ignore")
        filtered_dq_rows = [
            row
            for row in dq_rows
            if row.feature not in {self.time_column, self.target_column}
        ]
        cleaned_feature_df = clean_batch(
            feature_df,
            filtered_dq_rows,
            max_missing_rate=get_nested(
                self.config, "cleaning", "max_missing_rate", default=0.5
            ),
            min_unique_ratio=get_nested(
                self.config, "cleaning", "min_unique_ratio", default=0.10
            ),
            max_unique_ratio=get_nested(
                self.config, "cleaning", "max_unique_ratio", default=0.90
            ),
        )
        return pd.concat([df[protected_columns], cleaned_feature_df], axis=1)

    def resolve_feature_columns(
        self,
        df: pd.DataFrame,
    ) -> tuple[list[str], list[str], list[str]]:
        numeric_features, categorical_features = get_feature_columns(
            df,
            time_column=self.time_column,
            target_column=self.target_column,
            numeric_override=get_nested(
                self.config, "preprocessing", "numeric_features"
            ),
            categorical_override=get_nested(
                self.config, "preprocessing", "categorical_features"
            ),
        )
        feature_set = set(numeric_features) | set(categorical_features)
        feature_columns = [col for col in df.columns if col in feature_set]
        return feature_columns, numeric_features, categorical_features

    @abstractmethod
    def prepare(
        self,
        df: pd.DataFrame,
        *,
        dq_rows: list[DQRow] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError
