"""Campaign, channel and campaign x segment analytics (observational)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CAMPAIGNS, CAMPAIGN_LABELS, COST_PER_CONTACT, REVENUE_PER_RESPONSE, CHANNEL_COLS
from .evaluation import wilson_ci, adjust_pvalues, cramers_v, rate_table


def campaign_summary(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CAMPAIGNS:
        k, n = feat[c].sum(), len(feat)
        lo, hi = wilson_ci(k, n)
        acc = feat[feat[c] == 1]
        rows.append({"campaign": CAMPAIGN_LABELS[c], "column": c, "acceptors": int(k), "base": n,
                     "acceptance_rate": k / n, "ci_low": float(lo), "ci_high": float(hi),
                     "acceptor_avg_spend": acc["total_spend"].mean(),
                     "acceptor_median_income": acc["Income"].median(),
                     "acceptor_avg_recency": acc["Recency"].mean(),
                     "first_time_acceptors": int(((feat[c] == 1) & (feat[CAMPAIGNS[:CAMPAIGNS.index(c)]].sum(axis=1) == 0)).sum()) if c != CAMPAIGNS[0] else int(k)})
    return pd.DataFrame(rows)


def campaign_overlap(feat: pd.DataFrame) -> pd.DataFrame:
    """P(accept column | accepted row) — conditional acceptance between campaigns."""
    m = pd.DataFrame(index=[CAMPAIGN_LABELS[c] for c in CAMPAIGNS], columns=[CAMPAIGN_LABELS[c] for c in CAMPAIGNS], dtype=float)
    for a in CAMPAIGNS:
        for b in CAMPAIGNS:
            sub = feat[feat[a] == 1]
            m.loc[CAMPAIGN_LABELS[a], CAMPAIGN_LABELS[b]] = sub[b].mean() if len(sub) else np.nan
    return m


def repeat_response(feat: pd.DataFrame) -> pd.DataFrame:
    """Pilot response by number of campaigns (C1-C5) previously accepted."""
    t = rate_table(feat.assign(prior=feat["n_accepted_prior5"].clip(upper=3).astype(str).replace({"3": "3+"})),
                   "prior", "Response")
    return t.rename(columns={"prior": "prior_acceptances"})


def pilot_economics(feat: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    """Actual economics of the pilot (C6), using cost 3 MU/contact and revenue 11 MU/response
    from the dataset. Revenue here is campaign revenue per response as defined by the case,
    NOT customer spend and NOT profit margin."""
    def econ(d):
        n, k = len(d), d["Response"].sum()
        cost, rev = n * COST_PER_CONTACT, k * REVENUE_PER_RESPONSE
        lo, hi = wilson_ci(k, n)
        return pd.Series({"contacts": n, "responses": k, "response_rate": k / n, "rr_ci_low": lo, "rr_ci_high": hi,
                          "cost": cost, "revenue": rev, "profit": rev - cost,
                          "roi": (rev - cost) / cost, "profit_per_contact": (rev - cost) / n})
    if group is None:
        return econ(feat).to_frame("pilot_all_customers").T
    return feat.groupby(group, observed=True).apply(econ, include_groups=False).sort_values("profit_per_contact", ascending=False)


def campaign_by_segment(feat: pd.DataFrame, seg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Acceptance rate matrix (segment x campaign) + per-campaign chi-square with BH correction."""
    rates = feat.groupby(seg, observed=True)[CAMPAIGNS].mean().rename(columns=CAMPAIGN_LABELS)
    idx = rates.mean(axis=0)
    index = rates / idx  # index vs overall rate of that campaign (1.0 = average)
    tests = []
    for c in CAMPAIGNS:
        tab = pd.crosstab(feat[seg], feat[c])
        chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
        tests.append({"campaign": CAMPAIGN_LABELS[c], "chi2": chi2, "dof": dof, "p_value": p,
                      "cramers_v": cramers_v(tab)})
    tests = pd.DataFrame(tests)
    tests["p_bh"] = adjust_pvalues(tests["p_value"], "fdr_bh")
    return rates, index, tests


def segment_campaign_long(feat: pd.DataFrame, seg: str) -> pd.DataFrame:
    rows = []
    for s, d in feat.groupby(seg, observed=True):
        for c in CAMPAIGNS:
            k, n = d[c].sum(), len(d)
            lo, hi = wilson_ci(k, n)
            rows.append({"segment": s, "campaign": CAMPAIGN_LABELS[c], "n": n, "acceptors": int(k),
                         "rate": k / n, "ci_low": float(lo), "ci_high": float(hi),
                         "overall_rate": feat[c].mean()})
    out = pd.DataFrame(rows)
    out["index_vs_overall"] = out["rate"] / out["overall_rate"]
    return out


def channel_summary(feat: pd.DataFrame) -> pd.DataFrame:
    """Purchase channels (web/catalog/store) — these are where customers BUY, not the channel
    through which campaigns were delivered (unobserved). Spend is not split by channel."""
    rows = []
    tot = feat[list(CHANNEL_COLS.values())].sum().sum()
    for name, col in CHANNEL_COLS.items():
        users = feat[feat[col] > 0]
        dom = feat[feat["dominant_channel"] == name]
        k, n = dom["Response"].sum(), len(dom)
        lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append({"channel": name, "purchases": int(feat[col].sum()), "purchase_share": feat[col].sum() / tot,
                     "customers_using": len(users), "pct_customers_using": len(users) / len(feat),
                     "avg_spend_users": users["total_spend"].mean(),
                     "dominant_customers": n, "dominant_avg_spend": dom["total_spend"].mean() if n else np.nan,
                     "dominant_pilot_response": k / n if n else np.nan, "dom_ci_low": lo, "dom_ci_high": hi})
    return pd.DataFrame(rows)


def channel_intensity_response(feat: pd.DataFrame) -> pd.DataFrame:
    """Pilot response and spend by purchase-count tercile in each channel (volume vs value)."""
    rows = []
    for name, col in CHANNEL_COLS.items():
        q = pd.qcut(feat[col].rank(method="first"), 3, labels=["low", "mid", "high"])
        for lvl, d in feat.groupby(q, observed=True):
            k, n = d["Response"].sum(), len(d)
            lo, hi = wilson_ci(k, n)
            rows.append({"channel": name, "usage_tercile": lvl, "n": n,
                         "purchases_range": f"{d[col].min()}-{d[col].max()}",
                         "avg_total_spend": d["total_spend"].mean(), "pilot_response": k / n,
                         "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def funnel(feat: pd.DataFrame) -> pd.DataFrame:
    """Only the stages the data observes for the pilot: contacted -> accepted. Web visits are
    'last month' (timing vs campaign unknown) so they are NOT used as a funnel stage."""
    n, k = len(feat), feat["Response"].sum()
    return pd.DataFrame([{"stage": "Contacted in pilot (C6)", "customers": n, "pct_of_contacted": 1.0},
                         {"stage": "Accepted pilot offer", "customers": int(k), "pct_of_contacted": k / n}])
