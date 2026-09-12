"""Documented, defensible cleaning. Raw data are never modified; every step is logged."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAMPAIGNS


def clean_ifood(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log = []

    def note(step, n, detail):
        log.append({"step": step, "rows_affected": int(n), "detail": detail})

    note("start", len(df), "raw rows")
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    note("parse Dt_Customer", len(df), "ISO date -> datetime")

    # 0) category normalisation FIRST, so records differing only by a label variant are caught as duplicates
    mapping = {"Alone": "Single", "Absurd": "Unknown", "YOLO": "Unknown"}
    n = df["Marital_Status"].isin(mapping).sum()
    df["Marital_Status"] = df["Marital_Status"].replace(mapping)
    note("Marital_Status normalisation", n, "Alone->Single; Absurd/YOLO->Unknown")

    # 1) exact duplicates except ID (same profile, same outcome) -> keep lowest ID
    df = df.sort_values("ID")
    dup_mask = df.drop(columns="ID").duplicated(keep="first")
    note("drop exact duplicates (all fields but ID)", dup_mask.sum(),
         "identical on 28 fields incl. spend, dates, outcomes -> duplicate registrations")
    df = df[~dup_mask]

    # 2) groups identical on every feature but with conflicting Response -> exclude
    feats = [c for c in df.columns if c not in ("ID", "Response")]
    conflict = df.duplicated(subset=feats, keep=False)
    note("exclude conflicting-label duplicate groups", conflict.sum(),
         "same customer profile recorded with both Response=0 and 1; true label unknowable")
    df = df[~conflict]

    # 3) implausible values -> NaN + flag (never silently dropped)
    df["income_invalid_flag"] = (df["Income"] > 200_000).astype(int)
    note("Income 666,666 placeholder -> NaN", df["income_invalid_flag"].sum(), "flagged")
    df.loc[df["income_invalid_flag"] == 1, "Income"] = np.nan
    df["income_missing_flag"] = df["Income"].isna().astype(int)
    note("Income missing (incl. invalid)", df["income_missing_flag"].sum(),
         "kept as NaN in clean table; imputed inside modelling pipelines only")

    df["birth_year_invalid_flag"] = (df["Year_Birth"] < 1920).astype(int)
    note("Year_Birth < 1920 -> age NaN", df["birth_year_invalid_flag"].sum(), "flagged")

    # 5) constants become parameters, not features
    df = df.drop(columns=["Z_CostContact", "Z_Revenue"])
    note("drop constant Z_ columns", 0, "Z_CostContact=3, Z_Revenue=11 moved to config economics")

    for c in CAMPAIGNS:
        df[c] = df[c].astype(int)
    note("final", len(df), "analytical customer rows")
    return df.reset_index(drop=True), pd.DataFrame(log)


def clean_hillstrom(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    log = [{"step": "start", "rows_affected": len(df), "detail": "raw rows"}]
    n = (df.zip_code == "Surburban").sum()
    df["zip_code"] = df["zip_code"].replace({"Surburban": "Suburban"})
    log.append({"step": "zip_code typo", "rows_affected": int(n), "detail": "Surburban -> Suburban"})
    df["treatment"] = df["segment"].map({"No E-Mail": "Control", "Mens E-Mail": "Mens", "Womens E-Mail": "Womens"})
    df["treated"] = (df["treatment"] != "Control").astype(int)
    df["history_band"] = df["history_segment"].str.slice(3)
    df["recency_band"] = pd.cut(df["recency"], [0, 3, 6, 9, 12], labels=["1-3m", "4-6m", "7-9m", "10-12m"])
    df["buyer_type"] = np.select([(df.mens == 1) & (df.womens == 1), df.mens == 1, df.womens == 1],
                                 ["Both", "Mens only", "Womens only"], "None")
    df.insert(0, "customer_idx", np.arange(len(df)))
    log.append({"step": "derive", "rows_affected": len(df),
                "detail": "treatment label, history_band, recency_band, buyer_type, customer_idx (row id)"})
    log.append({"step": "final", "rows_affected": len(df), "detail": "no rows removed"})
    return df, pd.DataFrame(log)
