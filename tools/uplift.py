"""Incremental (causal) response on the Hillstrom randomised e-mail experiment.

Randomisation (1/3 Mens e-mail, 1/3 Womens e-mail, 1/3 no e-mail) makes arm-vs-control
differences unbiased estimates of the average treatment effect in this population. Subgroup
effects use only PRE-treatment covariates and are corrected for multiple testing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from .config import SEED
from .evaluation import two_prop_diff, mean_diff_ci, adjust_pvalues, wilson_ci

ARMS = ["Mens", "Womens"]
OUTCOMES = ["visit", "conversion", "spend"]
SUBGROUP_DIMS = ["history_band", "recency_band", "buyer_type", "newbie", "channel", "zip_code"]
COVARIATES_NUM = ["recency", "history", "mens", "womens", "newbie"]
COVARIATES_CAT = ["zip_code", "channel"]


def arm_summary(h: pd.DataFrame) -> pd.DataFrame:
    g = h.groupby("treatment")
    out = pd.DataFrame({"customers": g.size(), "visit_rate": g.visit.mean(), "conversion_rate": g.conversion.mean(),
                        "spend_per_customer": g.spend.mean(), "spend_per_converter": h[h.conversion == 1].groupby("treatment").spend.mean()})
    lo, hi = wilson_ci(g.conversion.sum(), g.size())
    out["conv_ci_low"], out["conv_ci_high"] = lo, hi
    return out.reset_index()


def effect(d: pd.DataFrame, arm: str, outcome: str) -> dict:
    t, c = d[d.treatment == arm][outcome], d[d.treatment == "Control"][outcome]
    if outcome == "spend":
        est, lo, hi, p = mean_diff_ci(t, c)
    else:
        est, lo, hi, p = two_prop_diff(t.sum(), len(t), c.sum(), len(c))
    base = c.mean()
    return {"arm": arm, "outcome": outcome, "n_treated": len(t), "n_control": len(c), "control_mean": base,
            "treated_mean": t.mean(), "effect": est, "ci_low": lo, "ci_high": hi, "p_value": p,
            "relative_lift": est / base if base > 0 else np.nan}


def average_effects(h: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([effect(h, a, o) for a in ARMS for o in OUTCOMES])
    # Mens vs Womens head-to-head
    for o in OUTCOMES:
        m, w = h[h.treatment == "Mens"][o], h[h.treatment == "Womens"][o]
        est, lo, hi, p = (mean_diff_ci(m, w) if o == "spend" else two_prop_diff(m.sum(), len(m), w.sum(), len(w)))
        out = pd.concat([out, pd.DataFrame([{"arm": "Mens minus Womens", "outcome": o, "n_treated": len(m),
                                             "n_control": len(w), "control_mean": w.mean(), "treated_mean": m.mean(),
                                             "effect": est, "ci_low": lo, "ci_high": hi, "p_value": p,
                                             "relative_lift": est / w.mean()}])], ignore_index=True)
    out["p_holm"] = adjust_pvalues(out["p_value"], "holm")
    return out


def subgroup_effects(h: pd.DataFrame, outcomes=("conversion", "spend")) -> pd.DataFrame:
    rows = []
    for dim in SUBGROUP_DIMS:
        for lvl, d in h.groupby(dim, observed=True):
            for a in ARMS:
                for o in outcomes:
                    r = effect(d, a, o)
                    rows.append({"dimension": dim, "level": str(lvl), **r})
    out = pd.DataFrame(rows)
    out["p_bh"] = adjust_pvalues(out["p_value"], "fdr_bh")
    out["significant_bh_5pct"] = out["p_bh"] < 0.05
    return out


def heterogeneity_tests(sub: pd.DataFrame) -> pd.DataFrame:
    """Cochran's Q test of equal effects across levels of each dimension (per arm x outcome)."""
    rows = []
    for (dim, arm, o), d in sub.groupby(["dimension", "arm", "outcome"]):
        se = (d.ci_high - d.ci_low) / (2 * 1.959964)
        w = 1 / se**2
        pooled = (w * d.effect).sum() / w.sum()
        Q = (w * (d.effect - pooled) ** 2).sum()
        df_ = len(d) - 1
        rows.append({"dimension": dim, "arm": arm, "outcome": o, "levels": len(d), "Q": Q, "df": df_,
                     "p_value": 1 - stats.chi2.cdf(Q, df_),
                     "effect_min": d.effect.min(), "effect_max": d.effect.max()})
    out = pd.DataFrame(rows)
    out["p_bh"] = adjust_pvalues(out["p_value"], "fdr_bh")
    return out


def _clf():
    pre = ColumnTransformer([("num", StandardScaler(), COVARIATES_NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), COVARIATES_CAT)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def t_learner(train: pd.DataFrame, outcome: str = "conversion") -> dict:
    """One outcome model per arm (T-learner). Predicted uplift = P(y|arm,x) - P(y|control,x)."""
    return {arm: _clf().fit(train[train.treatment == arm][COVARIATES_NUM + COVARIATES_CAT],
                            train[train.treatment == arm][outcome])
            for arm in ["Control"] + ARMS}


def predict_uplift(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    p = {a: m.predict_proba(X[COVARIATES_NUM + COVARIATES_CAT])[:, 1] for a, m in models.items()}
    return pd.DataFrame({"p_control": p["Control"], "p_mens": p["Mens"], "p_womens": p["Womens"],
                         "uplift_mens": p["Mens"] - p["Control"], "uplift_womens": p["Womens"] - p["Control"]},
                        index=X.index)


def qini_curve(test: pd.DataFrame, score: np.ndarray, arm: str, outcome: str = "conversion", bins: int = 20) -> pd.DataFrame:
    """Cumulative incremental outcome when targeting the top fraction by score, estimated from
    randomised arm vs control within each top fraction (scaled to the treated count)."""
    d = test[test.treatment.isin([arm, "Control"])].assign(score=score[test.treatment.isin([arm, "Control"]).values])
    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    t = (d.treatment == arm).values; y = d[outcome].values
    nt, nc = np.cumsum(t), np.cumsum(~t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * ~t)
    idx = np.unique(np.linspace(0, len(d) - 1, bins + 1).astype(int))[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        inc = yt[idx] - yc[idx] * nt[idx] / np.maximum(nc[idx], 1)
    frac = (idx + 1) / len(d)
    return pd.DataFrame({"share_targeted": np.r_[0, frac], "incremental": np.r_[0, inc]})


def qini_coefficient(curve: pd.DataFrame) -> float:
    x, y = curve.share_targeted.values, curve.incremental.values
    random_line = y[-1] * x
    return float(np.trapezoid(y - random_line, x))


def policy_value(test: pd.DataFrame, assignment: np.ndarray, outcome: str = "spend") -> dict:
    """Unbiased estimate of mean outcome if the policy were applied, using randomised arms:
    average outcome over customers whose actual arm equals the policy's arm, reweighted by
    the empirical arm propensity (IPW)."""
    prop = test.treatment.value_counts(normalize=True)
    match = test.treatment.values == assignment
    w = 1 / prop.reindex(test.treatment.values).values
    val = (test[outcome].values * match * w).sum() / len(test)
    return {"value_per_customer": val, "emails_per_customer": float((assignment != "Control").mean())}


def run(h: pd.DataFrame) -> dict:
    train, test = train_test_split(h, test_size=0.5, random_state=SEED, stratify=h.treatment)
    res = {"arm_summary": arm_summary(h), "ate": average_effects(h)}
    sub = subgroup_effects(h)
    res["subgroups"] = sub
    res["heterogeneity"] = heterogeneity_tests(sub)

    # Uplift vs response ranking, for conversion (primary, rare) and visit (secondary, common).
    # Random-ranking reference = 100 random permutations -> mean and 95% band of the Qini coefficient.
    res["qini"], rows = {}, []
    rng = np.random.default_rng(SEED)
    up_by_outcome = {}
    for outcome in ["conversion", "visit"]:
        models = t_learner(train, outcome)
        up = predict_uplift(models, test)
        up_by_outcome[outcome] = up
        for arm, col, pcol in [("Mens", "uplift_mens", "p_mens"), ("Womens", "uplift_womens", "p_womens")]:
            rand = [qini_coefficient(qini_curve(test, rng.random(len(test)), arm, outcome)) for _ in range(100)]
            for name, sc in [("uplift (T-learner)", up[col].values), ("response model P(y|email)", up[pcol].values)]:
                cur = qini_curve(test, sc, arm, outcome)
                q = qini_coefficient(cur)
                res["qini"][(outcome, arm, name)] = cur
                rows.append({"outcome": outcome, "arm": arm, "ranking": name, "qini_coefficient": q,
                             "random_mean": np.mean(rand), "random_p2_5": np.percentile(rand, 2.5),
                             "random_p97_5": np.percentile(rand, 97.5),
                             "beats_random_95": q > np.percentile(rand, 97.5),
                             "total_incremental_in_arm": cur.incremental.iloc[-1]})
    res["qini_table"] = pd.DataFrame(rows)
    up = up_by_outcome["conversion"]
    models = t_learner(train, "conversion")

    # Policies (evaluated on held-out half with IPW, spend = revenue in $, not margin)
    # Segment-rule policy learned on TRAIN: best arm per buyer_type x history tercile by spend effect
    train = train.assign(hist_t=pd.qcut(train.history, 3, labels=["low", "mid", "high"]))
    edges = train.history.quantile([1/3, 2/3]).values
    test = test.assign(hist_t=pd.cut(test.history, [-np.inf, *edges, np.inf], labels=["low", "mid", "high"]))
    rule = {}
    for key, d in train.groupby(["buyer_type", "hist_t"], observed=True):
        eff = {a: d[d.treatment == a].spend.mean() - d[d.treatment == "Control"].spend.mean() for a in ARMS}
        best = max(eff, key=eff.get)
        rule[key] = best if eff[best] > 0 else "Control"
    res["segment_rule"] = pd.DataFrame([{"buyer_type": k[0], "history_tercile": k[1], "assigned_email": v}
                                        for k, v in rule.items()])
    seg_assign = np.array([rule.get((b, t), "Mens") for b, t in zip(test.buyer_type, test.hist_t)])
    best_model = np.where(up[["uplift_mens", "uplift_womens"]].max(axis=1) > 0,
                          np.where(up.uplift_mens >= up.uplift_womens, "Mens", "Womens"), "Control")
    policies = {"No e-mail": np.full(len(test), "Control"), "E-mail all: Mens": np.full(len(test), "Mens"),
                "E-mail all: Womens": np.full(len(test), "Womens"),
                "Segment rule (buyer type x history)": seg_assign,
                "Uplift model: best arm per customer": best_model}
    rows = []
    for name, a in policies.items():
        s = policy_value(test, a, "spend"); c = policy_value(test, a, "conversion")
        rows.append({"policy": name, "spend_per_customer": s["value_per_customer"],
                     "conversion_rate": c["value_per_customer"], "emails_per_customer": s["emails_per_customer"]})
    pol = pd.DataFrame(rows)
    base = pol.loc[pol.policy == "No e-mail", "spend_per_customer"].iloc[0]
    pol["incremental_spend_per_customer"] = pol["spend_per_customer"] - base
    pol["incremental_spend_per_email"] = pol["incremental_spend_per_customer"] / pol["emails_per_customer"].replace(0, np.nan)
    res["policies"] = pol
    # bootstrap CI for policy values (test-set resampling)
    rng = np.random.default_rng(SEED); boots = {k: [] for k in policies}
    tv = test.reset_index(drop=True)
    for _ in range(300):
        i = rng.integers(0, len(tv), len(tv)); tb = tv.iloc[i]
        b0 = policy_value(tb, policies["No e-mail"][i])["value_per_customer"]
        for k, a in policies.items():
            boots[k].append(policy_value(tb, a[i])["value_per_customer"] - b0)
    res["policies"]["inc_spend_ci_low"] = [np.percentile(boots[k], 2.5) for k in policies]
    res["policies"]["inc_spend_ci_high"] = [np.percentile(boots[k], 97.5) for k in policies]
    res["test_uplift"] = pd.concat([test.reset_index(drop=True), up.reset_index(drop=True)], axis=1)
    res["models"] = models
    return res


def email_cost_scenarios(pol: pd.DataFrame, costs=(0.0, 0.05, 0.10, 0.25, 0.50, 1.00)) -> pd.DataFrame:
    """HYPOTHETICAL: the dataset has no e-mail cost and spend is revenue, not margin. Shows
    net incremental revenue per 1,000 customers under assumed per-email costs."""
    rows = []
    for c in costs:
        for r in pol.itertuples():
            rows.append({"assumed_cost_per_email": c, "policy": r.policy,
                         "net_incremental_per_1000_customers": 1000 * (r.incremental_spend_per_customer - c * r.emails_per_customer)})
    return pd.DataFrame(rows)
