from src.analysis.data_quality import DQRow
from src.preprocessing.preprocessors import (
    CatBoostPreparedData,
    CatBoostPreprocessor,
    MLPPreparedData,
    MLPPreprocessor,
    Preprocessor,
    extract_target,
)
from src.preprocessing.transformers import build_preprocessor, get_feature_columns


def prepare_features_for_mlp(
    df,
    config,
    *,
    fit: bool = False,
    preprocessor=None,
    dq_rows: list[DQRow] | None = None,
) -> MLPPreparedData:
    return MLPPreprocessor(config).prepare(
        df,
        fit=fit,
        preprocessor=preprocessor,
        dq_rows=dq_rows,
    )


def prepare_features_for_catboost(
    df,
    config,
    *,
    dq_rows: list[DQRow] | None = None,
) -> CatBoostPreparedData:
    return CatBoostPreprocessor(config).prepare(
        df,
        dq_rows=dq_rows,
    )


__all__ = [
    "CatBoostPreparedData",
    "CatBoostPreprocessor",
    "MLPPreparedData",
    "MLPPreprocessor",
    "Preprocessor",
    "build_preprocessor",
    "extract_target",
    "get_feature_columns",
    "prepare_features_for_catboost",
    "prepare_features_for_mlp",
]
