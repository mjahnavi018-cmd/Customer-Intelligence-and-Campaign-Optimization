"""Data-quality audit: generic column profile + dataset-specific rule checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MNT_COLS, CAMPAIGNS


def column_profile(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "dataset": dataset, "column": c, "dtype": str(s.dtype),
            "n_rows": len(s), "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2), "n_unique": int(s.nunique()),
            "min": s.min() if num else None, "median": s.median() if num else None,
            "max": s.max() if num else None,
            "top_value": s.value_counts().index[0] if s.notna().any() else None,
            "top_share_pct": round(100 * s.value_counts(normalize=True).iloc[0], 2),
        })
    return pd.DataFrame(rows)


def _iqr_outliers(s: pd.Series, k: float = 3.0) -> int:
    q1, q3 = s.quantile([.25, .75])
    return int(((s < q1 - k * (q3 - q1)) | (s > q3 + k * (q3 - q1))).sum())


def ifood_rule_checks(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in df.columns if c not in ("ID", "Response")]
    dt = pd.to_datetime(df["Dt_Customer"], errors="coerce")
    purchases = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
    checks = [
        ("duplicate IDs", int(df["ID"].duplicated().sum()), "none"),
        ("rows identical except ID (incl. Response)", int(df.drop(columns="ID").duplicated().sum()),
         "duplicate registrations -> remove extra copies"),
        ("rows identical except ID & Response (conflicting label groups)",
         int(df.duplicated(subset=feats).sum() - df.drop(columns="ID").duplicated().sum()),
         "same profile, contradictory Response -> exclude group"),
        ("Income missing", int(df["Income"].isna().sum()), "impute (median) + missing flag"),
        ("Income > 200,000 (implausible placeholder 666,666)", int((df["Income"] > 200_000).sum()),
         "set to missing + flag"),
        ("Income extreme outliers (3xIQR)", _iqr_outliers(df["Income"].dropna()), "reviewed"),
        ("Year_Birth < 1920 (age > 94 at 2014)", int((df["Year_Birth"] < 1920).sum()),
         "set age to missing + flag"),
        ("Marital_Status invalid ('Absurd','YOLO')", int(df["Marital_Status"].isin(["Absurd", "YOLO"]).sum()),
         "map to 'Unknown'"),
        ("Marital_Status 'Alone'", int((df["Marital_Status"] == "Alone").sum()), "map to 'Single'"),
        ("Education '2n Cycle' vs 'Master' (ambiguous coding)", int((df["Education"] == "2n Cycle").sum()),
         "kept as-is; documented"),
        ("Dt_Customer unparseable", int(dt.isna().sum()), "none"),
        ("Dt_Customer range", f"{dt.min().date()} .. {dt.max().date()}", "tenure reference = max+1 day"),
        ("Z_CostContact constant", df["Z_CostContact"].nunique() == 1, "used as cost/contact = 3 MU"),
        ("Z_Revenue constant", df["Z_Revenue"].nunique() == 1, "used as revenue/response = 11 MU"),
        ("customers with 0 web+catalog+store purchases", int((purchases == 0).sum()), "flag"),
        ("NumDealsPurchases > total channel purchases", int((df["NumDealsPurchases"] > purchases).sum()),
         "flag (deals are a subset of purchases)"),
        ("negative monetary values", int((df[MNT_COLS] < 0).sum().sum()), "none"),
        ("Recency distribution ~uniform 0-99 (possible synthetic)",
         round(float(np.histogram(df["Recency"], bins=10)[0].std() / np.histogram(df["Recency"], bins=10)[0].mean()), 3),
         "coefficient of variation of decile counts; documented as data-provenance caveat"),
        ("Complain = 1", int(df["Complain"].sum()), "rare; kept"),
    ]
    for c in CAMPAIGNS:
        checks.append((f"{c} acceptance rate", round(float(df[c].mean()), 4), "class imbalance"))
    return pd.DataFrame(checks, columns=["check", "result", "treatment"]).assign(dataset="ifood")


def hillstrom_rule_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("spend > 0 but conversion = 0 (or vice versa)", int(((df.spend > 0) != (df.conversion == 1)).sum()), "none"),
        ("mens = womens = 0", int(((df.mens == 0) & (df.womens == 0)).sum()), "none"),
        ("zip_code typo 'Surburban'", int((df.zip_code == "Surburban").sum()), "rename to 'Suburban'"),
        ("fully identical rows", int(df.duplicated().sum()),
         "kept: 12 low-cardinality fields, no customer ID -> collisions expected"),
        ("treatment arm sizes", df.segment.value_counts().to_dict(), "balanced ~1/3 each (randomised)"),
        ("conversion rate", round(float(df.conversion.mean()), 4), "rare outcome (0.9%)"),
        ("visit rate", round(float(df.visit.mean()), 4), ""),
    ]
    return pd.DataFrame(checks, columns=["check", "result", "treatment"]).assign(dataset="hillstrom")


def randomisation_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Covariate balance across Hillstrom arms (standardised mean differences vs control)."""
    X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                       dtype=float)
    ctrl = X[df.segment == "No E-Mail"]
    rows = []
    for arm in ["Mens E-Mail", "Womens E-Mail"]:
        t = X[df.segment == arm]
        for c in X.columns:
            sd = np.sqrt((t[c].var() + ctrl[c].var()) / 2)
            rows.append({"arm": arm, "covariate": c, "mean_arm": t[c].mean(), "mean_control": ctrl[c].mean(),
                         "smd": (t[c].mean() - ctrl[c].mean()) / sd if sd > 0 else 0})
    return pd.DataFrame(rows)
