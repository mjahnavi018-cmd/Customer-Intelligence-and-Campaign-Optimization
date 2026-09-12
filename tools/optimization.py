"""Budget / resource allocation. iFood: back-tested on actual pilot outcomes (real cost & revenue
parameters). Everything forward-looking is labelled as model-expected under stated assumptions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST_PER_CONTACT, REVENUE_PER_RESPONSE, SEED


def budget_curve(y, scores: dict, budgets, cost=COST_PER_CONTACT, revenue=REVENUE_PER_RESPONSE) -> pd.DataFrame:
    y = np.asarray(y)
    rows = []
    for name, s in scores.items():
        if name.startswith("Random"):
            # expected value of random selection (exact, no simulation)
            for B in budgets:
                k = min(len(y), int(B // cost))
                r = k * y.mean()
                rows.append({"strategy": name, "budget_mu": B, "contacts": k, "responses": r,
                             "profit_mu": r * revenue - k * cost})
            continue
        order = np.argsort(-np.asarray(s, float), kind="mergesort")
        cum = np.concatenate([[0], np.cumsum(y[order])])
        for B in budgets:
            k = min(len(y), int(B // cost))
            rows.append({"strategy": name, "budget_mu": B, "contacts": k, "responses": cum[k],
                         "profit_mu": cum[k] * revenue - k * cost})
    out = pd.DataFrame(rows)
    out["roi"] = out["profit_mu"] / (out["contacts"] * cost).replace(0, np.nan)
    return out


def bootstrap_profit(y, score, contacts_list, n_boot=1000, cost=COST_PER_CONTACT, revenue=REVENUE_PER_RESPONSE, seed=SEED):
    """Customer-resampling bootstrap of back-tested profit when contacting the top-k share."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(score, float); n = len(y)
    rows = []
    for share in contacts_list:
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            k = int(round(share * n))
            o = np.argsort(-s[i], kind="mergesort")[:k]
            vals.append(y[i][o].sum() * revenue - k * cost)
        rows.append({"contact_share": share, "profit_mean": np.mean(vals),
                     "profit_ci_low": np.percentile(vals, 2.5), "profit_ci_high": np.percentile(vals, 97.5),
                     "p_profit_positive": np.mean(np.array(vals) > 0)})
    return pd.DataFrame(rows)


def marginal_returns(y, score, step_share=0.05, cost=COST_PER_CONTACT, revenue=REVENUE_PER_RESPONSE) -> pd.DataFrame:
    """Responses and profit gained by each additional 5% of customers contacted (in rank order).
    Diminishing returns here are OBSERVED from ranked outcomes, not assumed."""
    y = np.asarray(y); order = np.argsort(-np.asarray(score, float), kind="mergesort")
    n = len(y); rows = []
    edges = np.unique(np.round(np.arange(0, 1 + 1e-9, step_share) * n).astype(int))
    for a, b in zip(edges[:-1], edges[1:]):
        r = y[order[a:b]].sum()
        rows.append({"from_share": a / n, "to_share": b / n, "contacts": b - a, "responses": int(r),
                     "response_rate": r / (b - a), "marginal_profit_mu": r * revenue - (b - a) * cost})
    return pd.DataFrame(rows)


def economics_sensitivity(y, score, costs=(2.0, 3.0, 4.0), revenues=(8.0, 11.0, 14.0)) -> pd.DataFrame:
    y = np.asarray(y); order = np.argsort(-np.asarray(score, float), kind="mergesort")
    cum = np.cumsum(y[order]); k = np.arange(1, len(y) + 1)
    rows = []
    for c in costs:
        for r in revenues:
            prof = cum * r - k * c
            j = int(np.argmax(prof))
            rows.append({"cost_per_contact": c, "revenue_per_response": r, "break_even_p": c / r,
                         "hindsight_best_contacts": j + 1, "hindsight_best_share": (j + 1) / len(y),
                         "hindsight_best_profit": prof[j], "profit_contact_all": cum[-1] * r - len(y) * c,
                         "profit_top20pct": prof[int(0.2 * len(y)) - 1]})
    return pd.DataFrame(rows)


def scenario_table(y, scores: dict, feat_test: pd.DataFrame, p_raw_model: np.ndarray) -> pd.DataFrame:
    """Named allocation scenarios, all back-tested on the actual pilot outcome."""
    y = np.asarray(y); n = len(y)

    def run(name, mask, rule):
        k = int(mask.sum()); r = int(y[mask].sum())
        return {"scenario": name, "rule": rule, "contacts": k, "budget_used_mu": k * COST_PER_CONTACT,
                "responses": r, "response_rate": r / k if k else np.nan,
                "profit_mu": r * REVENUE_PER_RESPONSE - k * COST_PER_CONTACT,
                "share_of_all_responders_captured": r / y.sum()}

    top = lambda s, share: np.isin(np.arange(n), np.argsort(-np.asarray(s, float), kind="mergesort")[:int(round(share * n))])
    rows = [
        run("A  Mass contact (what the pilot did)", np.ones(n, bool), "contact every customer"),
        run("B  Historical-responder rule", scores["B1_prior_acceptances"] > 0,
            "contact customers who accepted >=1 of C1-C5"),
        run("C  RFM top 20%", top(scores["B2_RFM_score"], .2), "highest R+F+M scores"),
        run("D  Value-focused top 20%", top(feat_test["total_spend"].values, .2), "highest 2-yr spend"),
        run("E  Model top 20%", top(p_raw_model, .2), "highest predicted response (temporal LR)"),
        run("F  Model ex-ante break-even rule", p_raw_model >= COST_PER_CONTACT / REVENUE_PER_RESPONSE,
            "contact if model p >= 0.273 (model calibrated to C2-C5 base rates)"),
    ]
    return pd.DataFrame(rows)
