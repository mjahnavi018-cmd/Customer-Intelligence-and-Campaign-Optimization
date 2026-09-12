"""Historical customer value. NOTE: this is HISTORICAL value (2-year category spend), not CLV.
The snapshot has no transaction timestamps, so future value / CLV cannot be estimated defensibly."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import gini_from_values, rate_table


def lorenz(values) -> pd.DataFrame:
    v = np.sort(np.asarray(values, float))[::-1]  # descending: top customers first
    cum = np.cumsum(v) / v.sum()
    share_cust = np.arange(1, len(v) + 1) / len(v)
    return pd.DataFrame({"share_customers": share_cust, "share_revenue": cum})


def concentration_summary(feat: pd.DataFrame) -> pd.DataFrame:
    v = feat["total_spend"]
    lz = lorenz(v)
    rows = [{"metric": "Gini coefficient of 2-year spend", "value": gini_from_values(v)}]
    for p in (0.01, 0.05, 0.10, 0.20, 0.50):
        rows.append({"metric": f"revenue share of top {int(p*100)}% customers",
                     "value": float(lz.loc[lz.share_customers <= p + 1e-12, "share_revenue"].iloc[-1])})
    rows.append({"metric": "customers needed for 80% of revenue (share)",
                 "value": float(lz.loc[lz.share_revenue >= 0.8, "share_customers"].iloc[0])})
    return pd.DataFrame(rows)


def value_deciles(feat: pd.DataFrame) -> pd.DataFrame:
    d = feat.assign(value_decile=10 - pd.qcut(feat["total_spend"].rank(method="first"), 10, labels=False))
    g = d.groupby("value_decile")
    out = pd.DataFrame({"customers": g.size(), "min_spend": g["total_spend"].min(), "max_spend": g["total_spend"].max(),
                        "revenue": g["total_spend"].sum(), "avg_purchases": g["total_purchases"].mean(),
                        "avg_aov": g["aov"].mean(), "prior5_accept_rate": g["n_accepted_prior5"].mean() / 5,
                        "pilot_response": g["Response"].mean(), "avg_income": g["Income"].mean()})
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    return out.reset_index()


def cohort_value(feat: pd.DataFrame) -> pd.DataFrame:
    """Enrolment cohort vs value. Enrolment cohort is the only 'acquisition' attribute available;
    the ACQUISITION CHANNEL IS NOT IN THE DATA. Spend per tenure month controls for exposure time."""
    g = feat.groupby("enrol_quarter")
    return pd.DataFrame({"customers": g.size(), "avg_tenure_months": g["tenure_months"].mean(),
                         "avg_total_spend": g["total_spend"].mean(), "avg_spend_per_month": g["spend_per_month"].mean(),
                         "median_spend_per_month": g["spend_per_month"].median(),
                         "pilot_response": g["Response"].mean(),
                         "prior5_accept_rate": g["n_accepted_prior5"].mean() / 5}).reset_index()


def value_vs_response(feat: pd.DataFrame) -> pd.DataFrame:
    d = feat.assign(value_tercile=pd.qcut(feat["total_spend"].rank(method="first"), 3, labels=["Low", "Mid", "High"]))
    return rate_table(d, "value_tercile", "Response")


def category_mix(feat: pd.DataFrame) -> pd.DataFrame:
    from .config import MNT_COLS
    tot = feat[MNT_COLS].sum()
    return pd.DataFrame({"category": [c.replace("Mnt", "") for c in MNT_COLS], "revenue": tot.values,
                         "share": (tot / tot.sum()).values,
                         "pct_customers_buying": [(feat[c] > 0).mean() for c in MNT_COLS]}).sort_values("revenue", ascending=False)
