from dataclasses import dataclass

import pandas as pd

from src.analysis.cleaning import CleaningPlan
from src.analysis.data_quality import DQRow
from src.preprocessing.preprocessors.interface import Preprocessor, extract_target


@dataclass(slots=True)
class CatBoostPreparedData:
    X: pd.DataFrame
    y: pd.Series
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_columns: list[str]


class CatBoostPreprocessor(Preprocessor):
    def prepare(
        self,
        df: pd.DataFrame,
        *,
        cleaning_plan: CleaningPlan | None = None,
        dq_rows: list[DQRow] | None = None,
    ) -> CatBoostPreparedData:
        prepared_df = self.apply_optional_cleaning(
            df,
            cleaning_plan=cleaning_plan,
            dq_rows=dq_rows,
        )
        y = extract_target(prepared_df, self.target_column)
        feature_columns, numeric_features, categorical_features = (
            self.resolve_feature_columns(prepared_df)
        )
        X = prepared_df[feature_columns].copy()
        for column in categorical_features:
            if column not in X.columns:
                continue
            X[column] = X[column].where(X[column].notna(), "missing").astype(str)
        dropped_columns = sorted(set(df.columns) - set(prepared_df.columns))

        return CatBoostPreparedData(
            X=X,
            y=y,
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            dropped_columns=dropped_columns,
        )
