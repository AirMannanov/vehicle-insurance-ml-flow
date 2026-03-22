from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer

from src.analysis.cleaning import CleaningPlan
from src.analysis.data_quality import DQRow
from src.preprocessing.preprocessors.interface import Preprocessor, extract_target
from src.preprocessing.transformers import (
    build_preprocessor,
    ensure_feature_columns,
    extract_preprocessor_feature_groups,
)
from src.tools import get_nested


@dataclass(slots=True)
class MLPPreparedData:
    X: object
    y: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    dropped_columns: list[str]


class MLPPreprocessor(Preprocessor):
    def prepare(
        self,
        df: pd.DataFrame,
        *,
        cleaning_plan: CleaningPlan | None = None,
        fit: bool = False,
        preprocessor: ColumnTransformer | None = None,
        dq_rows: list[DQRow] | None = None,
    ) -> MLPPreparedData:
        prepared_df = self.apply_optional_cleaning(
            df,
            cleaning_plan=cleaning_plan,
            dq_rows=dq_rows,
        )
        y = extract_target(prepared_df, self.target_column)
        dropped_columns = sorted(set(df.columns) - set(prepared_df.columns))

        if preprocessor is None:
            feature_columns, numeric_features, categorical_features = (
                self.resolve_feature_columns(prepared_df)
            )
            preprocessor = build_preprocessor(
                numeric_features,
                categorical_features,
                scaler=get_nested(
                    self.config, "preprocessing", "scaler", default="standard"
                ),
            )
            fit = True
        else:
            if not hasattr(preprocessor, "transformers_"):
                raise ValueError("Preprocessor must be fitted before fit=False transform")
            feature_columns, numeric_features, categorical_features = (
                extract_preprocessor_feature_groups(preprocessor)
            )
            prepared_df = ensure_feature_columns(
                prepared_df,
                numeric_features,
                categorical_features,
            )

        if fit:
            X = preprocessor.fit_transform(prepared_df)
        else:
            X = preprocessor.transform(prepared_df)
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        return MLPPreparedData(
            X=X,
            y=y,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            dropped_columns=dropped_columns,
        )
