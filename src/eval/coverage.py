"""Coverage evaluation: marginal and conditional-on-state slicing.

The central diagnostic of the paper: a method can be marginally valid
(coverage ~ 1-alpha on average over time) while badly miscovering
conditionally on market state. `coverage_by_state` quantifies that.
"""

import numpy as np
import pandas as pd


def marginal_coverage(df: pd.DataFrame, covered_col: str = "covered") -> float:
    d = df[~df.get("warmup", False)]
    return float(d[covered_col].mean())


def coverage_by_state(
    df: pd.DataFrame,
    state_col: str,
    bins: list[float] | int = (0.0, 0.5, 0.8, 0.95, 1.0),
    covered_col: str = "covered",
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Coverage sliced by quantile bins of a state variable (e.g. vix_pctl).

    Default bins: calm (<50th pctl), normal (50-80), elevated (80-95),
    stress (>95th pctl of the state variable).
    """
    d = df[~df.get("warmup", False)].copy()
    if labels is None and not isinstance(bins, int):
        labels = ["calm", "normal", "elevated", "stress"][: len(bins) - 1]
    d["state_bin"] = pd.cut(d[state_col], bins=bins, labels=labels,
                            include_lowest=True)
    out = d.groupby("state_bin", observed=True).agg(
        coverage=(covered_col, "mean"),
        upper_coverage=("covered_hi", "mean"),
        lower_coverage=("covered_lo", "mean"),
        n=(covered_col, "size"),
        mean_width=("width", "mean") if "width" in d.columns else (covered_col, "size"),
    )
    return out


def interval_width(df: pd.DataFrame) -> pd.Series:
    """Interval width in log-RV units (q_lo + q_hi)."""
    return df["q_lo"] + df["q_hi"]
