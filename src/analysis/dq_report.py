import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.database.connection import Database
from src.database.dataset_schema import COLUMN_TYPES, get_value_type
from src.data.storage import load_batch, list_batches

logger = logging.getLogger("mlops")


def _load_eda_data(db: Database) -> pd.DataFrame:
    """Load all batches from DB and concatenate for EDA section."""
    batches = list_batches(db)
    if not batches:
        raise ValueError("No batches in database.")
    dfs = []
    for row in batches:
        batch_id = row["batch_id"]
        df = load_batch(db, batch_id)
        df["_batch_id"] = batch_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _eda_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "_batch_id" and get_value_type(c) == "numeric"]


def _eda_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "_batch_id" and get_value_type(c) == "categorical"]


def _plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    num_cols = _eda_numeric_columns(df)
    if len(num_cols) < 2:
        return
    n = len(num_cols)
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
        origin="upper",
        extent=(-0.5, n - 0.5, n - 0.5, -0.5),
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)
    ax.set_xticks([i - 0.5 for i in range(n + 1)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(n + 1)], minor=True)
    ax.tick_params(axis="both", which="minor", length=0, labelbottom=False, labelleft=False)
    ax.grid(True, which="minor", linestyle="-", color="gray", linewidth=0.5)
    ax.grid(False, which="major")
    plt.colorbar(im, ax=ax, label="Correlation")
    plt.title("Correlation matrix (numeric features)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_numeric_histograms(df: pd.DataFrame, output_path: Path, max_cols: int = 12) -> None:
    num_cols = _eda_numeric_columns(df)[:max_cols]
    if not num_cols:
        return
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, col in enumerate(num_cols):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Numeric feature distributions", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _md_table_row(cells: list) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def _build_eda_md_content(df: pd.DataFrame, figures_rel: Path) -> str:
    """Build EDA section markdown: overview, numeric stats, correlation, categorical top-10."""
    num_cols = _eda_numeric_columns(df)
    cat_cols = _eda_categorical_columns(df)
    n_rows = len(df)
    n_cols = len([c for c in df.columns if c in COLUMN_TYPES])

    lines = [
        "# Automatic EDA",
        "",
        "## Summary",
        "",
        f"- **Rows:** {n_rows:,}",
        f"- **Columns:** {n_cols}",
        "",
        "## 1. Overview by column",
        "",
        "| Column | Type | Missing % | Unique count |",
        "|--------|------|-----------|---------------|",
    ]
    for col in df.columns:
        if col not in COLUMN_TYPES:
            continue
        s = df[col]
        missing_pct = 100.0 * s.isna().mean()
        n_unique = s.nunique()
        dtype = get_value_type(col)
        lines.append(_md_table_row([col, dtype, f"{missing_pct:.2f}%", n_unique]))
    lines.append("")

    if num_cols:
        lines.extend(["## 2. Numeric features", ""])
        desc = df[num_cols].describe().round(4)
        desc_df = desc.reset_index()
        header = list(desc_df.columns)
        lines.append(_md_table_row(header))
        lines.append("|" + "---|" * len(header))
        for _, row in desc_df.iterrows():
            lines.append(_md_table_row(list(row)))
        lines.append("")
        lines.append("### Correlation matrix")
        lines.append("")
        lines.append(f"![Correlation matrix]({figures_rel.as_posix()}/correlation_heatmap.png)")
        lines.append("")
        lines.append("### Distributions")
        lines.append("")
        lines.append(f"![Numeric distributions]({figures_rel.as_posix()}/numeric_histograms.png)")
        lines.append("")

    if cat_cols:
        lines.extend(["## 3. Categorical features (top 10 values)", ""])
        for col in cat_cols:
            vc = df[col].value_counts().head(10)
            lines.append(f"### {col}")
            lines.append("")
            lines.append("| value | count |")
            lines.append("|-------|-------|")
            for val, cnt in vc.items():
                lines.append(_md_table_row([val, cnt]))
            lines.append("")

    return "\n".join(lines)


def load_dq_rows(db: Database) -> list[dict]:
    return db.fetchall(
        """
        SELECT 
            batch_id,
            feature,
            missing_rate,
            unique_count,
            value_type,
            stats_json 
        FROM data_quality 
        ORDER BY batch_id, feature
        """
    )


def _batch_aggregates(by_batch: dict[str, list[dict]], batches: list[str]) -> tuple[list[float], list[float]]:
    """Return (avg_missing_rates, avg_unique_counts) per batch in batch order."""
    missing_rates = []
    unique_counts = []
    for batch_id in batches:
        br = by_batch[batch_id]
        avg_miss = sum(r["missing_rate"] for r in br) / len(br)
        avg_uniq = sum(r["unique_count"] for r in br) / len(br)
        missing_rates.append(avg_miss)
        unique_counts.append(avg_uniq)
    return missing_rates, unique_counts


def plot_batch_dq(
    by_batch: dict[str, list[dict]],
    batches: list[str],
    output_path: Path,
) -> None:
    missing_rates, unique_counts = _batch_aggregates(by_batch, batches)
    n = len(batches)
    x = list(range(n))

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except (OSError, ValueError):
            continue

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    step = max(1, n // 12)
    tick_indices = x[::step]
    tick_labels = [batches[i] for i in tick_indices]

    # Top: missing_rate
    ax1.plot(x, missing_rates, color="#1f77b4", linewidth=2, label="avg missing_rate")
    ax1.set_ylabel("avg missing_rate", fontsize=11)
    ax1.set_ylim(0, max(max(missing_rates) * 1.1, 0.01))
    ax1.tick_params(axis="y", labelsize=9)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    # Bottom: unique_count
    ax2.plot(x, unique_counts, color="#d62728", linewidth=2, label="avg unique_count")
    ax2.set_xlabel("Batch (month)", fontsize=11)
    ax2.set_ylabel("avg unique_count", fontsize=11)
    ax2.tick_params(axis="both", labelsize=9)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_report(rows: list[dict], chart_filename: str | None = None) -> str:
    if not rows:
        return "# Data Quality Report\n\nNo data quality records in the database.\n"

    batches = sorted({r["batch_id"] for r in rows})
    features = sorted({r["feature"] for r in rows})
    by_batch: dict[str, list[dict]] = {}
    by_feature: dict[str, list[dict]] = {}
    for r in rows:
        by_batch.setdefault(r["batch_id"], []).append(r)
        by_feature.setdefault(r["feature"], []).append(r)

    lines = [
        "# Data Quality Report",
        "",
        "## Summary",
        "",
        f"- **Batches:** {len(batches)}",
        f"- **Features:** {len(features)}",
        f"- **Records:** {len(rows)}",
        "",
        "---",
        "",
        "## 1. By batch (aggregates)",
        "",
    ]
    if chart_filename:
        lines.append(f"![missing_rate and unique_count by batch]({chart_filename})")
        lines.append("")
    else:
        lines.extend([
            "| batch_id | n_features | avg missing_rate | avg unique_count |",
            "|----------|------------|------------------|------------------|",
        ])
        for batch_id in batches:
            br = by_batch[batch_id]
            avg_miss = sum(r["missing_rate"] for r in br) / len(br)
            avg_uniq = sum(r["unique_count"] for r in br) / len(br)
            lines.append(f"| {batch_id} | {len(br)} | {avg_miss:.4f} | {avg_uniq:.1f} |")
        lines.append("")
    lines.extend([
        "",
        "## 2. By feature (aggregates)",
        "",
        "| feature | value_type | n_batches | avg missing_rate | min | max | avg unique_count | min | max |",
        "|---------|------------|-----------|------------------|-----|-----|------------------|-----|-----|",
    ])
    for feature in features:
        fr = by_feature[feature]
        rates = [r["missing_rate"] for r in fr]
        uniqs = [r["unique_count"] for r in fr]
        vt = fr[0]["value_type"] if fr else "—"
        lines.append(
            f"| {feature} | {vt} | {len(fr)} | {sum(rates)/len(rates):.4f} | {min(rates):.4f} | {max(rates):.4f} | "
            f"{sum(uniqs)/len(uniqs):.1f} | {min(uniqs)} | {max(uniqs)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(db: Database, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_dq_rows(db)
    chart_filename: str | None = None
    if rows:
        batches = sorted({r["batch_id"] for r in rows})
        by_batch: dict[str, list[dict]] = {}
        for r in rows:
            by_batch.setdefault(r["batch_id"], []).append(r)
        relative_chart = Path("figures") / "dq_chart.png"
        chart_path = output_path.parent / relative_chart
        plot_batch_dq(by_batch, batches, chart_path)
        chart_filename = str(relative_chart)
    dq_content = build_report(rows, chart_filename=chart_filename)

    try:
        df = _load_eda_data(db)
        figures_dir = output_path.parent / Path("figures") / "eda"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figures_rel = Path("figures") / "eda"
        _plot_correlation_heatmap(df, figures_dir / "correlation_heatmap.png")
        _plot_numeric_histograms(df, figures_dir / "numeric_histograms.png")
        eda_content = _build_eda_md_content(df, figures_rel)
        report = dq_content + "\n\n---\n\n" + eda_content
        logger.info("EDA: included in report (%d rows)", len(df))
    except Exception as e:
        logger.debug("EDA section skipped: %s", e)
        report = dq_content

    output_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s (%d DQ records)", output_path, len(rows))
    return output_path
