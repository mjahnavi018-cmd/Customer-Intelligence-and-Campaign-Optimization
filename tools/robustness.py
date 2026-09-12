"""Robustness / sensitivity checks. Each returns a table; the pipeline classifies findings as
STABLE or SENSITIVE using explicit criteria documented in reports/methodology.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, roc_auc_score, average_precision_score

from . import rfm as rfm_mod
from .config import SEED, CAMPAIGNS
from .evaluation import cramers_v, ranking_metrics
from .response_model import make_lr, FEATURE_SETS
from .segmentation import fit_kmeans


def rfm_binning(feat: pd.DataFrame, bins=(3, 4, 5, 10)) -> pd.DataFrame:
    ref = rfm_mod.rfm_segments(rfm_mod.rfm_table(feat, 5))
    rows = []
    for b in bins:
        r = rfm_mod.rfm_segments(rfm_mod.rfm_table(feat, b))
        hv_ref = set(ref.loc[ref.value_tier == "High value", "ID"]); hv = set(r.loc[r.value_tier == "High value", "ID"])
        y = feat.set_index("ID").loc[r.ID, "Response"].values
        top = r.RFM_score.rank(method="first", ascending=False) <= 0.2 * len(r)
        rows.append({"n_bins": b, "segment_ARI_vs_quintiles": adjusted_rand_score(ref.rfm_segment, r.rfm_segment),
                     "high_value_jaccard_vs_quintiles": len(hv & hv_ref) / len(hv | hv_ref),
                     "rfm_score_auc_on_pilot": roc_auc_score(y, r.RFM_score),
                     "pilot_response_top20pct_by_rfm": y[top.values].mean()})
    return pd.DataFrame(rows)


def kmeans_k_sensitivity(feat: pd.DataFrame, X: pd.DataFrame, k_star: int, ks=(3, 4, 5, 6)) -> pd.DataFrame:
    ref, _ = fit_kmeans(X, k_star)
    from sklearn.preprocessing import StandardScaler
    rows = []
    for k in ks:
        km, sc = fit_kmeans(X, k)
        lab = km.labels_
        rows.append({"k": k, "ARI_vs_selected": adjusted_rand_score(ref.labels_, lab),
                     "response_cramers_v": cramers_v(pd.crosstab(lab, feat["Response"])),
                     "response_rate_range": pd.Series(feat["Response"].values).groupby(lab).mean().agg(np.ptp)})
    return pd.DataFrame(rows)


def training_window(panel: pd.DataFrame, C: float) -> pd.DataFrame:
    fs = FEATURE_SETS["full"]
    test = panel[panel.campaign_k == 6]
    rows = []
    for name, ks in {"C5 only": [5], "C4-C5": [4, 5], "C2-C5 (primary)": [2, 3, 4, 5], "C2-C5 excl. C3": [2, 4, 5]}.items():
        tr = panel[panel.campaign_k.isin(ks)]
        m = make_lr(fs, C).fit(tr[fs], tr["target"])
        s = m.predict_proba(test[fs])[:, 1]
        rows.append({"training_campaigns": name, "train_rows": len(tr), **ranking_metrics(test["target"], s)})
    return pd.DataFrame(rows)


def per_campaign_generalisation(panel: pd.DataFrame, C: float) -> pd.DataFrame:
    """Train on all other observed campaigns (incl. C6), test on each campaign: performance drift."""
    fs = FEATURE_SETS["full"]
    rows = []
    for k in range(2, 7):
        tr, te = panel[panel.campaign_k != k], panel[panel.campaign_k == k]
        m = make_lr(fs, C).fit(tr[fs], tr["target"])
        s = m.predict_proba(te[fs])[:, 1]
        rows.append({"held_out_campaign": f"C{k}", "acceptance_rate": te.target.mean(),
                     "roc_auc": roc_auc_score(te.target, s), "pr_auc": average_precision_score(te.target, s),
                     "rule_auc_prior_acceptances": roc_auc_score(te.target, te.hist_n_prior + 0.5 * te.hist_last)})
    return pd.DataFrame(rows)


def dedup_sensitivity(raw: pd.DataFrame, clean_feat: pd.DataFrame) -> pd.DataFrame:
    prior = [f"AcceptedCmp{i}" for i in range(1, 6)]
    rows = []
    for name, d in [("raw (2,240 incl. duplicates)", raw), ("clean (de-duplicated)", clean_feat)]:
        cnt = d[prior].sum(axis=1)
        rows.append({"table": name, "customers": len(d), "pilot_response_rate": d["Response"].mean(),
                     "pilot_profit_mu": d["Response"].sum() * 11 - len(d) * 3,
                     "rule_auc_prior_acceptances": roc_auc_score(d["Response"], cnt + 0.5 * d["AcceptedCmp5"]),
                     "response_rate_if_prior_acceptor": d.loc[cnt > 0, "Response"].mean(),
                     "response_rate_if_no_prior": d.loc[cnt == 0, "Response"].mean()})
    return pd.DataFrame(rows)


def uplift_bootstrap(h: pd.DataFrame, n_boot=300) -> pd.DataFrame:
    """Stability of the key experimental conclusions under resampling of customers."""
    rng = np.random.default_rng(SEED)
    y_spend, arm = h.spend.values, h.treatment.values
    mo = (h.buyer_type == "Mens only").values
    conv = h.conversion.values
    out = {"mens_gt_womens_spend": [], "womens_gt_zero_spend": [], "womens_on_mens_only_conv_gt0": [],
           "mens_on_mens_only_conv_gt0": []}
    n = len(h)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, s, c, m = arm[i], y_spend[i], conv[i], mo[i]
        ctrl = s[a == "Control"].mean()
        out["mens_gt_womens_spend"].append(s[a == "Mens"].mean() > s[a == "Womens"].mean())
        out["womens_gt_zero_spend"].append(s[a == "Womens"].mean() > ctrl)
        cm = c[(a == "Control") & m].mean()
        out["womens_on_mens_only_conv_gt0"].append(c[(a == "Womens") & m].mean() > cm)
        out["mens_on_mens_only_conv_gt0"].append(c[(a == "Mens") & m].mean() > cm)
    return pd.DataFrame([{"conclusion": k, "share_of_bootstrap_resamples_true": float(np.mean(v))} for k, v in out.items()])
