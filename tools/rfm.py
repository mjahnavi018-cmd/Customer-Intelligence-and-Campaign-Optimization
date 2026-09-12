"""RFM scoring. Recency = days since last purchase (lower is better), Frequency = web+catalog+store
purchases, Monetary = total spend on 6 categories over the last 2 years."""
from __future__ import annotations

import numpy as np
import pandas as pd


def score(series: pd.Series, n_bins: int, higher_is_better: bool = True) -> pd.Series:
    """Quantile score 1..n_bins. Ties are broken by rank(method='first') so every bin has
    ~equal size even though Frequency has many tied integer values; the tie-break order is
    the (deterministic) row order. Documented as a limitation (tied customers may land in
    adjacent bins)."""
    s = series if higher_is_better else -series
    r = s.rank(method="first")
    return pd.qcut(r, n_bins, labels=False).astype(int) + 1


def rfm_table(feat: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    r = pd.DataFrame({"ID": feat["ID"], "recency_days": feat["Recency"],
                      "frequency": feat["total_purchases"], "monetary": feat["total_spend"]})
    r["R"] = score(r["recency_days"], n_bins, higher_is_better=False)
    r["F"] = score(r["frequency"], n_bins)
    r["M"] = score(r["monetary"], n_bins)
    r["RFM_score"] = r[["R", "F", "M"]].sum(axis=1)
    r["FM"] = (r["F"] + r["M"]) / 2
    return r


def rfm_segments(r: pd.DataFrame) -> pd.DataFrame:
    """Interpretable 3x3 grid: activity tier from Recency terciles x value tier from FM.
    Labels describe the scores themselves, so they are true by construction; the profile table
    (segment_profiles) is then used to check what they actually mean behaviourally."""
    out = r.copy()
    out["activity_tier"] = pd.qcut(out["recency_days"].rank(method="first"), 3,
                                   labels=["Recent", "Mid", "Lapsing"])
    # value tier: terciles of FM average
    out["value_tier"] = pd.qcut(out["FM"].rank(method="first"), 3, labels=["Low value", "Mid value", "High value"])
    names = {
        ("High value", "Recent"): "Core high-value (recent)",
        ("High value", "Mid"): "High-value (cooling)",
        ("High value", "Lapsing"): "High-value (lapsing)",
        ("Mid value", "Recent"): "Mid-value (recent)",
        ("Mid value", "Mid"): "Mid-value (cooling)",
        ("Mid value", "Lapsing"): "Mid-value (lapsing)",
        ("Low value", "Recent"): "Low-value (recent)",
        ("Low value", "Mid"): "Low-value (cooling)",
        ("Low value", "Lapsing"): "Low-value (lapsing)",
    }
    out["rfm_segment"] = [names[(v, a)] for v, a in zip(out["value_tier"], out["activity_tier"])]
    return out


def recency_tier_bounds(r: pd.DataFrame) -> pd.DataFrame:
    return r.groupby("activity_tier", observed=True)["recency_days"].agg(["min", "max", "count"])


def segment_profiles(df: pd.DataFrame, seg_col: str) -> pd.DataFrame:
    g = df.groupby(seg_col, observed=True)
    prof = pd.DataFrame({
        "customers": g.size(),
        "share_customers_pct": 100 * g.size() / len(df),
        "total_spend": g["total_spend"].sum(),
        "share_revenue_pct": 100 * g["total_spend"].sum() / df["total_spend"].sum(),
        "avg_spend": g["total_spend"].mean(),
        "median_spend": g["total_spend"].median(),
        "avg_purchases": g["total_purchases"].mean(),
        "avg_recency_days": g["Recency"].mean(),
        "avg_aov": g["aov"].mean(),
        "avg_web_visits": g["NumWebVisitsMonth"].mean(),
        "deals_share": g["deals_share"].mean(),
        "catalog_share": g["share_catalog"].mean(),
        "prior5_accept_rate": g["n_accepted_prior5"].mean() / 5,
        "pilot_response_rate": g["Response"].mean(),
        "avg_income": g["Income"].mean(),
        "avg_children": g["children"].mean(),
    })
    return prof.round(3)
