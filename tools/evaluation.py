"""Statistical helpers: intervals, tests, multiple-testing correction, ranking metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

from .config import SEED


def wilson_ci(k, n, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion (vectorised)."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(df: pd.DataFrame, group: str, target: str) -> pd.DataFrame:
    g = df.groupby(group, observed=True)[target].agg(["sum", "count"])
    g.columns = ["responders", "n"]
    g["rate"] = g["responders"] / g["n"]
    lo, hi = wilson_ci(g["responders"], g["n"])
    g["ci_low"], g["ci_high"] = lo, hi
    return g.reset_index()


def two_prop_diff(k1, n1, k0, n0, alpha=0.05):
    """Difference p1-p0 with Wald CI and two-sided z-test p-value (pooled)."""
    p1, p0 = k1 / n1, k0 / n0
    d = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = stats.norm.ppf(1 - alpha / 2)
    pp = (k1 + k0) / (n1 + n0)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    pval = 2 * (1 - stats.norm.cdf(abs(d) / se0)) if se0 > 0 else np.nan
    return d, d - z * se, d + z * se, pval


def mean_diff_ci(x1, x0, alpha=0.05):
    """Welch difference in means with CI and p-value."""
    x1, x0 = np.asarray(x1, float), np.asarray(x0, float)
    d = x1.mean() - x0.mean()
    se = np.sqrt(x1.var(ddof=1) / len(x1) + x0.var(ddof=1) / len(x0))
    z = stats.norm.ppf(1 - alpha / 2)
    p = stats.ttest_ind(x1, x0, equal_var=False).pvalue
    return d, d - z * se, d + z * se, p


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def chi2_test(df, group, target):
    tab = pd.crosstab(df[group], df[target])
    chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
    return {"chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v(tab), "n": int(tab.values.sum())}


def adjust_pvalues(p, method="fdr_bh"):
    """Benjamini-Hochberg ('fdr_bh') or Holm ('holm') adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    if method == "fdr_bh":
        adj = ranked * m / (np.arange(m) + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
    elif method == "holm":
        adj = ranked * (m - np.arange(m))
        adj = np.maximum.accumulate(adj)
    else:
        raise ValueError(method)
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


def lift_at(y, score, frac):
    y = np.asarray(y); score = np.asarray(score, float)
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-score, kind="mergesort")[:k]
    return y[idx].mean() / y.mean()


def ranking_metrics(y, score, fracs=(0.1, 0.2, 0.3)) -> dict:
    y = np.asarray(y); score = np.asarray(score, float)
    out = {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score)}
    for f in fracs:
        out[f"lift@{int(f*100)}%"] = lift_at(y, score, f)
        k = int(round(f * len(y)))
        idx = np.argsort(-score, kind="mergesort")[:k]
        out[f"recall@{int(f*100)}%"] = y[idx].sum() / y.sum()
    return out


def threshold_metrics(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0)}


def bootstrap_auc_diff(y, s1, s0, n_boot=1000, seed=SEED):
    """Paired bootstrap of ROC-AUC(s1) - ROC-AUC(s0) and PR-AUC difference."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s1 = np.asarray(s1, float); s0 = np.asarray(s0, float)
    d_roc, d_pr = [], []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d_roc.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s0[i]))
        d_pr.append(average_precision_score(y[i], s1[i]) - average_precision_score(y[i], s0[i]))
    q = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"roc_auc_diff": q(d_roc), "pr_auc_diff": q(d_pr)}


def bootstrap_metric_ci(y, s, fn, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s, float)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def gini_from_values(v) -> float:
    v = np.sort(np.asarray(v, float))
    n = len(v)
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
