"""Targeting engine: one transparent, evidence-backed row per customer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BREAK_EVEN_P, COST_PER_CONTACT, REVENUE_PER_RESPONSE

PRIORITY_RULES = {
    "P1 - Target": "accepted >= 1 of the observed campaigns (back-test on pilot: 40% response vs 27.3% "
                   "break-even). Ordered by model probability when budget is smaller than the tier.",
    "P2 - Test cell only": "never accepted, but in the top 10% of model probability among never-acceptors "
                           "(back-test: ~15% response, BELOW break-even -> only as a randomised test)",
    "P3 - Suppress from paid offer": "never accepted and not in the non-acceptor top decile",
}


def priority(n_accepted: np.ndarray, p: np.ndarray, nonacc_cut: float) -> np.ndarray:
    """Hybrid rule chosen from the back-test: campaign history defines the tiers (the model did not
    beat it significantly), the model orders customers within tiers."""
    n_accepted = np.asarray(n_accepted); p = np.asarray(p)
    return np.select([n_accepted > 0, p >= nonacc_cut],
                     ["P1 - Target", "P2 - Test cell only"], "P3 - Suppress from paid offer")


def nonacceptor_cut(n_accepted, p, q=0.9) -> float:
    p = np.asarray(p); return float(np.quantile(p[np.asarray(n_accepted) == 0], q))


def action(prio: str, value_tier: str, activity_tier: str) -> str:
    if prio.startswith("P1"):
        return ("Include in next offer - priority (responsive + high value)" if value_tier == "High value"
                else "Include in next offer (responsive; lower basket value)")
    if prio.startswith("P2"):
        return "Randomised test cell with holdout (expected below break-even)"
    if value_tier == "High value":
        return "No paid offer; high value but never accepted - review offer relevance / non-paid touch"
    return "Suppress from paid offer (no response history, low predicted response)"


def priority_backtest(y, n_prior, p_model) -> pd.DataFrame:
    """Apply the SAME tier definition to the pilot (history = C1-C5, model trained on C2-C5)."""
    cut = nonacceptor_cut(n_prior, p_model)
    tiers = priority(n_prior, p_model, cut)
    from .evaluation import wilson_ci
    rows = []
    for t in sorted(set(tiers)):
        m = tiers == t; k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"tier": t, "customers": n, "responders": k, "response_rate": k / n, "ci_low": float(lo),
                     "ci_high": float(hi), "profit_mu": k * REVENUE_PER_RESPONSE - n * COST_PER_CONTACT,
                     "above_break_even": k / n > BREAK_EVEN_P})
    return pd.DataFrame(rows)


def lr_contributions(model, X: pd.DataFrame, features) -> pd.DataFrame:
    """Per-customer log-odds contributions = standardised value x coefficient (exact for LR)."""
    pre = model.named_steps["pre"]
    Z = pre.transform(X[features])
    coef = model.named_steps["clf"].coef_[0]
    return pd.DataFrame(Z * coef, columns=features, index=X.index)


PRETTY = {"hist_n_prior": "campaign acceptances so far", "hist_last": "accepted most recent campaign",
          "hist_ever": "ever accepted a campaign", "hist_rate_prior": "historical acceptance rate",
          "total_spend": "2-yr spend", "Income": "income", "Recency": "days since last purchase",
          "share_catalog": "catalog purchase share", "share_gold": "gold-product share",
          "share_wines": "wine share", "share_meat": "meat share", "share_fish": "fish share",
          "Teenhome": "teenagers at home", "Kidhome": "young children at home", "children": "children at home",
          "category_entropy": "category breadth", "NumWebVisitsMonth": "web visits last month",
          "deals_share": "deal-purchase share", "tenure_days": "tenure", "share_store": "store share",
          "share_web": "web share", "aov": "average order value", "total_purchases": "purchases",
          "age": "age", "partnered": "partnered", "Complain": "complained",
          "income_missing_flag": "income missing", "share_sweet": "sweets share",
          "edu_basic": "basic education", "edu_2n_cycle": "2n-cycle education", "edu_master": "master's",
          "edu_phd": "PhD"}


def build_engine(feat: pd.DataFrame, scoring: pd.DataFrame, p_forward: np.ndarray,
                 contrib: pd.DataFrame, channel_evidence: pd.DataFrame) -> pd.DataFrame:
    e = feat[["ID", "rfm_segment", "cluster_label", "value_tier", "activity_tier", "R", "F", "M", "RFM_score",
              "total_spend", "total_purchases", "aov", "Recency", "tenure_months", "Income", "age",
              "NumWebVisitsMonth", "dominant_channel", "share_web", "share_catalog", "share_store",
              "n_accepted_prior5", "n_accepted_all6", "Response"] +
             [f"AcceptedCmp{i}" for i in range(1, 6)]].copy()
    e["p_next_campaign"] = p_forward
    e["expected_profit_mu"] = p_forward * REVENUE_PER_RESPONSE - COST_PER_CONTACT
    cut = nonacceptor_cut(e["n_accepted_all6"], p_forward)
    e["priority"] = priority(e["n_accepted_all6"], p_forward, cut)
    e["action"] = [action(p, v, a) for p, v, a in zip(e["priority"], e["value_tier"], e["activity_tier"])]
    tier_order = e["priority"].str.slice(1, 2).astype(int)
    e["priority_rank"] = (tier_order * 10 - e["p_next_campaign"]).rank(method="first").astype(int)
    e["value_pct_rank"] = e["total_spend"].rank(pct=True)

    ch = channel_evidence.set_index("group")
    e["channel_evidence"] = [
        (f"{d}: pilot response {ch.loc[d, 'rate']:.1%} (95% CI {ch.loc[d, 'ci_low']:.1%}-{ch.loc[d, 'ci_high']:.1%}, n={int(ch.loc[d, 'n'])})"
         if d in ch.index else d) for d in e["dominant_channel"]]

    reasons_pos, reasons_neg = [], []
    for i in range(len(e)):
        c = contrib.iloc[i].sort_values()
        pos = [f"{PRETTY.get(k, k)} (+{v:.2f})" for k, v in c[::-1].items() if v > 0.05][:3]
        neg = [f"{PRETTY.get(k, k)} ({v:.2f})" for k, v in c.items() if v < -0.05][:2]
        reasons_pos.append("; ".join(pos) if pos else "none material")
        reasons_neg.append("; ".join(neg) if neg else "none material")
    e["drivers_up_logodds"] = reasons_pos
    e["drivers_down_logodds"] = reasons_neg
    e["evidence_summary"] = [
        f"Accepted {int(r.n_accepted_all6)}/6 campaigns (pilot: {'yes' if r.Response else 'no'}); "
        f"2-yr spend {r.total_spend:,.0f} MU (top {100*(1-r.value_pct_rank):.0f}%); "
        f"last purchase {int(r.Recency)} days ago; model p(next)={r.p_next_campaign:.2f} (break-even {BREAK_EVEN_P:.2f})"
        for r in e.itertuples()]
    return e.sort_values("priority_rank").reset_index(drop=True)


def action_matrix(engine: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(engine["value_tier"], engine["priority"], margins=True)


def compare_priority_methods(test_df: pd.DataFrame, y: np.ndarray, methods: dict,
                             budgets_share=(0.1, 0.2, 0.3)) -> pd.DataFrame:
    """Backtest alternative prioritisation scores on the pilot campaign (actual outcomes).
    Reports responses captured (volume), profit and historical spend of captured responders (quality)."""
    rows = []
    n = len(y)
    for name, s in methods.items():
        order = np.argsort(-np.asarray(s, float), kind="mergesort")
        for b in budgets_share:
            k = int(round(b * n))
            idx = order[:k]
            resp = y[idx]
            rows.append({"method": name, "contact_share": b, "contacts": k, "responses": int(resp.sum()),
                         "response_rate": resp.mean(), "profit_mu": resp.sum() * REVENUE_PER_RESPONSE - k * COST_PER_CONTACT,
                         "avg_hist_spend_contacted": test_df["total_spend"].values[idx].mean(),
                         "avg_hist_spend_responders": test_df["total_spend"].values[idx][resp == 1].mean() if resp.sum() else np.nan})
    return pd.DataFrame(rows)


def case_studies(engine: pd.DataFrame, cxs_long: pd.DataFrame, backtest: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Pick REAL customers for each case type by explicit, reproducible rules; the markdown is
    generated from their actual fields."""
    e = engine
    picks = []
    hv = e[(e.priority.str.startswith("P1")) & (e.value_tier == "High value")]
    picks.append(("Case 1 - High-value customer", "highest 2-yr spend among P1 customers", hv.nlargest(1, "total_spend")))
    hp = e[(e.priority.str.startswith("P1")) & (e.value_tier != "High value")]
    picks.append(("Case 2 - High-response customer outside the RFM high-value tier", "highest model p among P1 customers whose RFM value tier is not High",
                  hp.nlargest(1, "p_next_campaign")))
    ar = e[(e.value_tier == "High value") & (e.activity_tier == "Lapsing") & (e.n_accepted_all6 > 0)]
    picks.append(("Case 3 - At-risk (high value, lapsing) customer", "high value tier, lapsing recency tier, has accepted before; highest spend",
                  ar.nlargest(1, "total_spend")))
    dis = e[(e.n_accepted_all6 == 0)]
    picks.append(("Case 4 - Model and rule disagree", "never accepted any campaign (rule says suppress) but highest model p",
                  dis.nlargest(1, "p_next_campaign")))
    rows, md = [], []
    for title, rule, d in picks:
        if d.empty:
            continue
        r = d.iloc[0]
        hist = "".join(str(int(r[f"AcceptedCmp{i}"])) for i in range(1, 6)) + str(int(r.Response))
        rows.append({"case": title, "selection_rule": rule, "ID": int(r.ID)})
        tier_bt = backtest.set_index("tier")
        bt = tier_bt.loc[r.priority] if r.priority in tier_bt.index else None
        md.append(f"### {title} — customer ID {int(r.ID)}\n"
                  f"*Selected by rule:* {rule}.\n\n"
                  f"1. **Profile:** income {r.Income:,.0f}, age {r.age:.0f}, tenure {r.tenure_months:.1f} months.\n"
                  f"2. **Behaviour:** {int(r.total_purchases)} purchases (web {r.share_web:.0%}, catalog {r.share_catalog:.0%}, store {r.share_store:.0%}); "
                  f"last purchase {int(r.Recency)} days ago; {int(r.NumWebVisitsMonth)} web visits last month.\n"
                  f"3. **Segment:** {r.rfm_segment} (RFM {int(r.R)}-{int(r.F)}-{int(r.M)}); cluster '{r.cluster_label}'.\n"
                  f"4. **Campaign history (C1..C6):** {hist} -> accepted {int(r.n_accepted_all6)}/6.\n"
                  f"5. **Value:** 2-yr spend {r.total_spend:,.0f} MU (percentile {100*r.value_pct_rank:.0f}); AOV {r.aov:,.1f} MU.\n"
                  f"6. **Response behaviour:** pilot (C6) {'accepted' if r.Response else 'did not accept'}.\n"
                  f"7. **Prediction:** model p(next campaign) = {r.p_next_campaign:.2f} vs break-even {BREAK_EVEN_P:.3f}; "
                  f"drivers up: {r.drivers_up_logodds}; drivers down: {r.drivers_down_logodds}.\n"
                  f"8. **Channel evidence:** {r.channel_evidence}. (Purchase channel, not campaign delivery channel.)\n"
                  f"9. **Recommendation:** {r.priority} -> {r.action}.\n"
                  f"10. **Why:** " + (f"customers in this tier responded at {bt.response_rate:.1%} (95% CI {bt.ci_low:.1%}-{bt.ci_high:.1%}) in the pilot back-test"
                                     if bt is not None else "see tier back-test") + ".\n"
                  f"11. **Limitation:** observational; response is not proof the offer *caused* a purchase; the pilot economics (3/11 MU) are assumed to carry over.\n")
    # Case 5: campaign x segment surprise — largest over-index among Low/Mid value segments (min 10 acceptors)
    c = cxs_long[(~cxs_long.segment.str.startswith("Core")) & (~cxs_long.segment.str.startswith("High")) & (cxs_long.acceptors >= 10)]
    s5 = c.sort_values("index_vs_overall", ascending=False).iloc[0]
    rows.append({"case": "Case 5 - Surprising campaign x segment", "selection_rule": "largest acceptance index among non-high-value segments (>=10 acceptors)",
                 "ID": None})
    md.append(f"### Case 5 — Surprising campaign × segment: {s5.campaign} in '{s5.segment}'\n"
              f"Campaign {s5.campaign} was accepted by {s5.rate:.1%} of '{s5.segment}' (95% CI {s5.ci_low:.1%}-{s5.ci_high:.1%}, "
              f"n={int(s5.n)}) vs {s5.overall_rate:.1%} overall (index {s5.index_vs_overall:.2f}), whereas most campaigns "
              f"over-index only in high-value segments. **Implication:** campaign content matters for who responds — a "
              f"C{s5.campaign[1]}-style offer is the only observed route to lower-value customers. **Limitation:** offer contents of "
              f"C1-C5 are not documented, so the *type* of offer cannot be identified; CI is wide.\n")
    return pd.DataFrame(rows), "\n".join(md)
