from typing import Literal

ValueType = Literal["numeric", "categorical"]

COLUMN_TYPES: dict[str, ValueType] = {
    "SEX": "categorical",
    "INSR_BEGIN": "categorical",
    "INSR_END": "categorical",
    "EFFECTIVE_YR": "categorical",
    "INSR_TYPE": "categorical",
    "INSURED_VALUE": "numeric",
    "PREMIUM": "numeric",
    "OBJECT_ID": "numeric",
    "PROD_YEAR": "numeric",
    "SEATS_NUM": "numeric",
    "CARRYING_CAPACITY": "numeric",
    "TYPE_VEHICLE": "categorical",
    "CCM_TON": "numeric",
    "MAKE": "categorical",
    "USAGE": "categorical",
    "CLAIM_PAID": "numeric",
}


def is_categorical(column: str) -> bool:
    if column not in COLUMN_TYPES:
        raise ValueError(f"Column {column} not found in COLUMN_TYPES")
    return COLUMN_TYPES[column] == "categorical"


def get_value_type(column: str) -> ValueType:
    if column not in COLUMN_TYPES:
        raise ValueError(f"Column {column} not found in COLUMN_TYPES")
    return COLUMN_TYPES[column]
