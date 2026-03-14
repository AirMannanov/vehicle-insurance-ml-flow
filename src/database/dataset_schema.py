from enum import StrEnum


class FeatureType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


_COLUMN_TYPES: dict[str, FeatureType] = {
    "SEX": FeatureType.CATEGORICAL,
    "INSR_BEGIN": FeatureType.CATEGORICAL,
    "INSR_END": FeatureType.CATEGORICAL,
    "EFFECTIVE_YR": FeatureType.CATEGORICAL,
    "INSR_TYPE": FeatureType.CATEGORICAL,
    "INSURED_VALUE": FeatureType.NUMERIC,
    "PREMIUM": FeatureType.NUMERIC,
    "OBJECT_ID": FeatureType.NUMERIC,
    "PROD_YEAR": FeatureType.NUMERIC,
    "SEATS_NUM": FeatureType.NUMERIC,
    "CARRYING_CAPACITY": FeatureType.NUMERIC,
    "TYPE_VEHICLE": FeatureType.CATEGORICAL,
    "CCM_TON": FeatureType.NUMERIC,
    "MAKE": FeatureType.CATEGORICAL,
    "USAGE": FeatureType.CATEGORICAL,
    "CLAIM_PAID": FeatureType.NUMERIC,
}


def column_in_schema(column: str) -> bool:
    return column in _COLUMN_TYPES


def get_feature_type(column: str) -> FeatureType:
    if column not in _COLUMN_TYPES:
        raise ValueError(f"Column {column} not found in dataset schema")
    return _COLUMN_TYPES[column]
