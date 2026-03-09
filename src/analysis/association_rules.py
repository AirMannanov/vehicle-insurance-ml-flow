import json
import logging
from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.database.connection import Database
from src.database.dataset_schema import get_value_type

logger = logging.getLogger("mlops")

# Fallback sequence for Apriori when too few rules are found (support, then confidence).
SUPPORT_LEVELS = [0.05, 0.03, 0.01]
CONFIDENCE_LEVELS = [0.3, 0.2, 0.1]
MIN_RULES_REQUIRED = 5


@dataclass
class AssocRule:
    """Single association rule with support, confidence, lift."""
    antecedents: list[str]
    consequents: list[str]
    support: float
    confidence: float
    lift: float


def _binarize_df(
    df: pd.DataFrame,
    exclude_columns: set[str],
) -> pd.DataFrame:
    """Convert batch DataFrame to binary matrix for Apriori.

    Uses dataset schema (get_value_type): categorical -> one-hot, numeric -> median split.
    Unknown columns are treated as numeric. Columns in exclude_columns are dropped.
    """
    out = []
    for col in df.columns:
        if col in exclude_columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        try:
            is_cat = get_value_type(col) == "categorical"
        except ValueError:
            is_cat = False
        if is_cat:
            dummies = pd.get_dummies(df[[col]], prefix=col, prefix_sep="=")
            dummies = (dummies == 1).astype(bool)
            out.append(dummies)
        else:
            median_val = df[col].median()
            above = (df[col] > median_val).fillna(False).astype(bool)
            above.name = f"{col}=above_median"
            out.append(pd.DataFrame({above.name: above}))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, axis=1)


def compute_assoc_rules(
    df: pd.DataFrame,
    exclude_columns: set[str] | None = None,
    min_rules: int = MIN_RULES_REQUIRED,
) -> list[AssocRule]:
    """Build binary matrix, run Apriori, return at least min_rules rules (by lift).

    Tries SUPPORT_LEVELS × CONFIDENCE_LEVELS until enough rules are found.
    """
    exclude_columns = exclude_columns or set()
    binary = _binarize_df(df, exclude_columns=exclude_columns)
    if binary.empty or binary.shape[1] < 2:
        logger.warning("Binarized matrix too small for association rules")
        return []

    rules_list: list[AssocRule] = []
    seen = set()
    for support in SUPPORT_LEVELS:
        for confidence in CONFIDENCE_LEVELS:
            try:
                frequent = apriori(binary, min_support=support, use_colnames=True)
                if frequent.empty:
                    continue
                rules_df = association_rules(
                    frequent, metric="confidence", min_threshold=confidence
                )
                if rules_df.empty:
                    continue
                for _, row in rules_df.iterrows():
                    ant = tuple(sorted(row["antecedents"]))
                    con = tuple(sorted(row["consequents"]))
                    key = (ant, con)
                    if key in seen:
                        continue
                    seen.add(key)
                    rules_list.append(
                        AssocRule(
                            antecedents=list(ant),
                            consequents=list(con),
                            support=float(row["support"]),
                            confidence=float(row["confidence"]),
                            lift=float(row["lift"]),
                        )
                    )
                if len(rules_list) >= min_rules:
                    break
            except Exception as e:
                logger.debug("Apriori attempt failed: %s", e)
                continue
        if len(rules_list) >= min_rules:
            break

    rules_list.sort(key=lambda r: r.lift, reverse=True)
    return rules_list


def save_assoc_rules(db: Database, batch_id: str, rules: list[AssocRule]) -> None:
    """Write association rules for one batch into assoc_rules table."""
    db.execute("DELETE FROM assoc_rules WHERE batch_id = ?", (batch_id,))
    for r in rules:
        db.execute(
            """INSERT INTO assoc_rules (batch_id, antecedents, consequents, support, confidence, lift)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                json.dumps(r.antecedents),
                json.dumps(r.consequents),
                r.support,
                r.confidence,
                r.lift,
            ),
        )
    db.commit()
    logger.debug("Saved %d association rules for batch %s", len(rules), batch_id)
