from dataclasses import dataclass

import pandas as pd

from src.analysis.data_quality import DQRow
from src.preprocessing.preprocessors.interface import Preprocessor, extract_target


@dataclass(slots=True)
class CatBoostPreparedData:
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


class CatBoostPreprocessor(Preprocessor):
    def prepare(
        self,
        df: pd.DataFrame,
        *,
        dq_rows: list[DQRow] | None = None,
    ) -> CatBoostPreparedData:
        prepared_df = self.apply_optional_cleaning(df, dq_rows=dq_rows)
        y = extract_target(prepared_df, self.target_column)
        feature_columns, numeric_features, categorical_features = (
            self.resolve_feature_columns(prepared_df)
        )
        X = prepared_df[feature_columns].copy()

        return CatBoostPreparedData(
            X=X,
            y=y,
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
