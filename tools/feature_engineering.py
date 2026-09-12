"""Customer-level features and the customer x campaign panel used for temporal validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MNT_COLS, CHANNEL_COLS, CAMPAIGNS

SNAPSHOT_YEAR = 2014  # last enrolment date is 2014-06-29; ages are computed at 2014


def customer_features(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    ref_date = df["Dt_Customer"].max() + pd.Timedelta(days=1)
    df["age"] = np.where(df["birth_year_invalid_flag"] == 1, np.nan, SNAPSHOT_YEAR - df["Year_Birth"])
    df["children"] = df["Kidhome"] + df["Teenhome"]
    df["tenure_days"] = (ref_date - df["Dt_Customer"]).dt.days
    df["tenure_months"] = df["tenure_days"] / 30.44
    df["enrol_quarter"] = df["Dt_Customer"].dt.to_period("Q").astype(str)

    df["total_spend"] = df[MNT_COLS].sum(axis=1)
    df["total_purchases"] = df[list(CHANNEL_COLS.values())].sum(axis=1)
    tp = df["total_purchases"].replace(0, np.nan)
    df["aov"] = df["total_spend"] / tp
    df["deals_share"] = (df["NumDealsPurchases"] / tp).clip(upper=1)
    for name, col in CHANNEL_COLS.items():
        df[f"share_{name.lower()}"] = df[col] / tp
    shares = df[[f"share_{n.lower()}" for n in CHANNEL_COLS]]
    dom = shares.fillna(0).idxmax(axis=1).str.replace("share_", "").str.title()
    df["dominant_channel"] = np.where(shares.max(axis=1) >= 0.5, dom, "Mixed")
    df.loc[df["total_purchases"] == 0, "dominant_channel"] = "No purchases"
    df["dominant_channel"] = df["dominant_channel"].astype(str)

    spend = df[MNT_COLS].div(df["total_spend"], axis=0)
    for c in MNT_COLS:
        df[f"share_{c.replace('Mnt', '').replace('Products', '').replace('Prods', '').lower()}"] = spend[c]
    ent = -(spend.where(spend > 0) * np.log(spend.where(spend > 0))).sum(axis=1)
    df["category_entropy"] = ent / np.log(len(MNT_COLS))
    df["spend_per_month"] = df["total_spend"] / df["tenure_months"].clip(lower=1)

    prior = [f"AcceptedCmp{i}" for i in range(1, 6)]
    df["n_accepted_prior5"] = df[prior].sum(axis=1)
    df["n_accepted_all6"] = df[CAMPAIGNS].sum(axis=1)
    df["ever_accepted_prior5"] = (df["n_accepted_prior5"] > 0).astype(int)
    return df


# Features that describe the customer at the snapshot (shared by every campaign row).
SNAPSHOT_FEATURES = ["Income", "income_missing_flag", "age", "children", "Kidhome", "Teenhome",
                     "tenure_days", "Recency", "total_spend", "total_purchases", "aov",
                     "deals_share", "share_web", "share_catalog", "share_store",
                     "NumWebVisitsMonth", "share_wines", "share_meat", "share_gold",
                     "share_fish", "share_sweet", "category_entropy", "Complain",
                     "edu_basic", "edu_2n_cycle", "edu_master", "edu_phd", "partnered"]
HISTORY_FEATURES = ["hist_n_prior", "hist_last", "hist_ever"]  # hist_rate_prior computed but excluded (collinear with n_prior)

# Feature groups used in leakage sensitivity tests.
TIMING_AMBIGUOUS = ["Recency"]
SPEND_BEHAVIOUR = ["total_spend", "total_purchases", "aov", "deals_share", "share_web",
                   "share_catalog", "share_store", "share_wines", "share_meat", "share_gold",
                   "share_fish", "share_sweet", "category_entropy", "NumWebVisitsMonth"]


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lvl in ["Basic", "2n Cycle", "Master", "PhD"]:
        out[f"edu_{lvl.lower().replace(' ', '_')}"] = (out["Education"] == lvl).astype(int)
    out["partnered"] = out["Marital_Status"].isin(["Married", "Together"]).astype(int)
    return out


def history_block(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Campaign-history features available BEFORE campaign k (1-indexed; k=7 = next campaign)."""
    prior = CAMPAIGNS[: k - 1]
    h = pd.DataFrame(index=df.index)
    h["hist_n_prior"] = df[prior].sum(axis=1)
    h["hist_rate_prior"] = h["hist_n_prior"] / len(prior)
    h["hist_last"] = df[prior[-1]]
    h["hist_ever"] = (h["hist_n_prior"] > 0).astype(int)
    return h


def campaign_panel(feat: pd.DataFrame, targets=range(2, 7)) -> pd.DataFrame:
    """Long table: one row per (customer, campaign k), target = accepted campaign k,
    history features use ONLY campaigns 1..k-1."""
    enc = _encode(feat)
    frames = []
    for k in targets:
        f = enc[["ID"] + SNAPSHOT_FEATURES].copy()
        f = pd.concat([f, history_block(enc, k)], axis=1)
        f["campaign_k"] = k
        f["target"] = enc[CAMPAIGNS[k - 1]].values
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def scoring_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Features for the NEXT (7th, not yet run) campaign: history over all six campaigns."""
    enc = _encode(feat)
    f = pd.concat([enc[["ID"] + SNAPSHOT_FEATURES], history_block(enc, 7)], axis=1)
    f["campaign_k"] = 7
    return f
