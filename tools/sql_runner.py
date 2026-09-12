"""Build the SQLite analytics database, execute sql/marketing_analytics.sql, save each result and
cross-check key numbers against the pandas pipeline.

Usage (after the Python pipeline has produced data/processed/):
    python -m src.sql_runner
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from .config import DB_PATH, SQL_DIR, TAB, DATA_PROCESSED, CAMPAIGNS, CAMPAIGN_LABELS


def build_db(feat: pd.DataFrame, hill: pd.DataFrame, engine: pd.DataFrame) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cust = pd.DataFrame({
        "ID": feat.ID, "total_spend": feat.total_spend, "total_purchases": feat.total_purchases,
        "web_purchases": feat.NumWebPurchases, "catalog_purchases": feat.NumCatalogPurchases,
        "store_purchases": feat.NumStorePurchases, "recency_days": feat.Recency, "income": feat.Income,
        "dt_customer": pd.to_datetime(feat.Dt_Customer).dt.strftime("%Y-%m-%d"), "rfm_segment": feat.rfm_segment,
        "value_tier": feat.value_tier, "activity_tier": feat.activity_tier, "cluster_label": feat.cluster_label,
        "dominant_channel": feat.dominant_channel, "response_c6": feat.Response})
    cust.to_sql("customers", con, index=False)
    long = feat.melt(id_vars="ID", value_vars=CAMPAIGNS, var_name="col", value_name="accepted")
    long["campaign_no"] = long.col.map({c: i + 1 for i, c in enumerate(CAMPAIGNS)})
    long["campaign_label"] = long.col.map(CAMPAIGN_LABELS)
    long[["ID", "campaign_no", "campaign_label", "accepted"]].to_sql("campaign_responses", con, index=False)
    hill[["customer_idx", "treatment", "buyer_type", "history", "recency", "channel", "zip_code", "newbie",
          "visit", "conversion", "spend"]].to_sql("hillstrom", con, index=False)
    engine[["ID", "p_next_campaign", "expected_profit_mu", "priority", "action"]].to_sql("targeting", con, index=False)
    con.execute("CREATE INDEX ix_cr ON campaign_responses(ID, campaign_no)")
    con.commit(); con.close()


def parse_queries(path=SQL_DIR / "marketing_analytics.sql") -> dict[str, str]:
    text = path.read_text()
    parts = re.split(r"^-- name: (\S+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def execute_all() -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(DB_PATH)
    out = {}
    for name, q in parse_queries().items():
        out[name] = pd.read_sql_query(q, con)
        out[name].to_csv(TAB / "sql" / f"{name}.csv", index=False)
    con.close()
    return out


def cross_check(res: dict, feat: pd.DataFrame, hill: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
    rows = []

    def chk(name, sql_val, py_val, tol=1e-6):
        rows.append({"check": name, "sql": float(sql_val), "python": float(py_val),
                     "abs_diff": abs(float(sql_val) - float(py_val)), "match": abs(float(sql_val) - float(py_val)) <= tol})

    q3 = res["q03_campaign_acceptance"]
    for i, c in enumerate(CAMPAIGNS):
        chk(f"acceptors {CAMPAIGN_LABELS[c]}", q3.acceptors.iloc[i], feat[c].sum())
    chk("customers", q3.customers.iloc[0], len(feat))
    q1 = res["q01_revenue_concentration_deciles"]
    chk("total revenue (2-yr spend)", q1.revenue.sum(), feat.total_spend.sum())
    n_top = int(q1.customers.iloc[:2].sum())  # NTILE sizes can differ by 1 -> compare like with like
    top20 = feat.total_spend.sort_values(ascending=False).head(n_top).sum() / feat.total_spend.sum()
    chk("top-2-decile revenue share", q1.cumulative_revenue_share.iloc[1], round(top20, 4), tol=1e-4)
    q7 = res["q07_pilot_economics_by_segment"].set_index("rfm_segment")
    py7 = feat.groupby("rfm_segment").apply(lambda d: d.Response.sum() * 11 - len(d) * 3, include_groups=False)
    for s in py7.index:
        chk(f"pilot profit {s}", q7.loc[s, "profit_mu"], py7[s])
    chk("pilot profit total", q7.profit_mu.sum(), feat.Response.sum() * 11 - len(feat) * 3)
    q12 = res["q12_hillstrom_arms"].set_index("treatment")
    for arm in ["Control", "Mens", "Womens"]:
        chk(f"hillstrom spend/customer {arm}", q12.loc[arm, "spend_per_customer"],
            round(hill.loc[hill.treatment == arm, "spend"].mean(), 4), tol=1e-4)
    q8 = res["q08_channel_volume_vs_value"].set_index("channel")
    chk("web purchases", q8.loc["Web", "purchases"], feat.NumWebPurchases.sum())
    q14 = res["q14_targeting_priority_summary"]
    chk("targeting rows", q14.customers.sum(), len(feat))
    if ctx is not None and "econ_seg" in ctx:
        es = ctx["econ_seg"].set_index("rfm_segment")
        chk("pilot profit (pipeline table) Core high-value", es.loc["Core high-value (recent)", "profit"],
            q7.loc["Core high-value (recent)", "profit_mu"])
    return pd.DataFrame(rows)


def run_all(ctx: dict | None = None) -> dict:
    if ctx is None:
        feat = pd.read_csv(DATA_PROCESSED / "customer_features.csv")
        hill = pd.read_csv(DATA_PROCESSED / "hillstrom_clean.csv")
        engine = pd.read_csv(DATA_PROCESSED / "targeting_engine.csv")
    else:
        feat, hill, engine = ctx["feat"], ctx["hill"], ctx["engine"]
    build_db(feat, hill, engine)
    res = execute_all()
    cc = cross_check(res, feat, hill, ctx)
    cc.to_csv(TAB / "sql_python_crosscheck.csv", index=False)
    print(f"SQL: {len(res)} queries executed; cross-checks passed {int(cc.match.sum())}/{len(cc)}")
    if not cc.match.all():
        print(cc[~cc.match])
        raise AssertionError("SQL/Python cross-check failed")
    return {"results": res, "crosscheck": cc}


if __name__ == "__main__":
    run_all()
