"""Response prediction with campaign-level temporal validation.

Design: customer x campaign panel. A model is trained on campaigns C2..C5 (target = accepted
campaign k, campaign history restricted to C1..k-1) and tested on the LATER pilot campaign C6
(target = Response) for all customers. Hyper-parameters are chosen by leave-one-campaign-out CV
inside the training campaigns — the test campaign is never used for any choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import brentq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, BREAK_EVEN_P
from .feature_engineering import (SNAPSHOT_FEATURES, HISTORY_FEATURES, TIMING_AMBIGUOUS,
                                  SPEND_BEHAVIOUR)
from .evaluation import ranking_metrics, bootstrap_auc_diff, bootstrap_metric_ci

SIMPLICITY_MARGIN = 0.01

FEATURE_SETS = {
    "full": SNAPSHOT_FEATURES + HISTORY_FEATURES,
    "no_recency": [f for f in SNAPSHOT_FEATURES if f not in TIMING_AMBIGUOUS] + HISTORY_FEATURES,
    "history_demographics_only": [f for f in SNAPSHOT_FEATURES
                                  if f not in TIMING_AMBIGUOUS + SPEND_BEHAVIOUR] + HISTORY_FEATURES,
}


def make_lr(features, C=1.0):
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("sc", StandardScaler())]), features)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=5000))])


def make_hgb(features, lr=0.05, depth=3, min_leaf=40):
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), features)])
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        learning_rate=lr, max_depth=depth, min_samples_leaf=min_leaf, max_iter=300,
        l2_regularization=1.0, early_stopping=False, random_state=SEED))])


def loco_cv(panel: pd.DataFrame, factory, features) -> float:
    """Leave-one-campaign-out mean PR-AUC across training campaigns."""
    scores = []
    for k in sorted(panel.campaign_k.unique()):
        tr, va = panel[panel.campaign_k != k], panel[panel.campaign_k == k]
        m = factory().fit(tr[features], tr["target"])
        scores.append(average_precision_score(va["target"], m.predict_proba(va[features])[:, 1]))
    return float(np.mean(scores))


def tune(panel_train: pd.DataFrame, features) -> pd.DataFrame:
    rows = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        rows.append({"model": "logistic_regression", "params": f"C={C}", "C": C,
                     "loco_pr_auc": loco_cv(panel_train, lambda C=C: make_lr(features, C), features)})
    for lr_, d, ml in [(0.03, 2, 40), (0.05, 3, 40), (0.05, 3, 100), (0.1, 2, 100)]:
        rows.append({"model": "gradient_boosting", "params": f"lr={lr_},depth={d},min_leaf={ml}",
                     "lr": lr_, "depth": d, "min_leaf": ml,
                     "loco_pr_auc": loco_cv(panel_train, lambda a=lr_, b=d, c=ml: make_hgb(features, a, b, c), features)})
    return pd.DataFrame(rows)


def best_factories(tuning: pd.DataFrame, features):
    lr_row = tuning[tuning.model == "logistic_regression"].sort_values("loco_pr_auc").iloc[-1]
    gb_row = tuning[tuning.model == "gradient_boosting"].sort_values("loco_pr_auc").iloc[-1]
    return {"logistic_regression": lambda: make_lr(features, lr_row.C),
            "gradient_boosting": lambda: make_hgb(features, gb_row.lr, int(gb_row.depth), int(gb_row.min_leaf))}


def baseline_scores(test: pd.DataFrame, rfm_score: pd.Series) -> dict:
    return {
        "B0_random_(overall_rate)": np.zeros(len(test)),
        "B1_prior_acceptances": test["hist_n_prior"].values + 0.5 * test["hist_last"].values,
        "B2_RFM_score": rfm_score.values.astype(float),
    }


def evaluate_scores(y, scores: dict) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        m = ranking_metrics(y, s)
        if np.ptp(s) > 0:
            m["roc_auc_ci"] = bootstrap_metric_ci(y, s, roc_auc_score, n_boot=500)
            m["pr_auc_ci"] = bootstrap_metric_ci(y, s, average_precision_score, n_boot=500)
        else:
            m["roc_auc_ci"] = (0.5, 0.5)
            m["pr_auc_ci"] = (y.mean(), y.mean())
        rows.append({"approach": name, **m})
    return pd.DataFrame(rows)


def prior_shift(p: np.ndarray, target_rate: float) -> np.ndarray:
    """Shift log-odds by a constant so mean(p) == target_rate (prior-shift / base-rate
    re-calibration). Preserves ranking."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    f = lambda d: expit(logit(p) + d).mean() - target_rate
    d = brentq(f, -10, 10)
    return expit(logit(p) + d)


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    return pd.DataFrame({"decile": q, "p": p, "y": y}).groupby("decile").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()


def threshold_curve(y, score, cost, revenue) -> pd.DataFrame:
    order = np.argsort(-score, kind="mergesort")
    ys = np.asarray(y)[order]
    n = len(ys)
    tp = np.cumsum(ys)
    k = np.arange(1, n + 1)
    prec = tp / k
    rec = tp / ys.sum()
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    profit = tp * revenue - k * cost
    return pd.DataFrame({"contacts": k, "share_contacted": k / n, "responses": tp, "precision": prec,
                         "recall": rec, "f1": f1, "profit": profit, "score_cutoff": np.asarray(score)[order]})


def logistic_coefficients(model, features) -> pd.DataFrame:
    coef = model.named_steps["clf"].coef_[0]
    return (pd.DataFrame({"feature": features, "std_coef": coef, "odds_ratio_per_sd": np.exp(coef)})
            .assign(abs=lambda d: d.std_coef.abs()).sort_values("abs", ascending=False).drop(columns="abs"))


def perm_importance(model, X, y, n_repeats=20) -> pd.DataFrame:
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats,
                               random_state=SEED)
    return (pd.DataFrame({"feature": X.columns, "pr_auc_drop_mean": r.importances_mean,
                          "pr_auc_drop_std": r.importances_std})
            .sort_values("pr_auc_drop_mean", ascending=False))


def run(panel: pd.DataFrame, rfm_by_id: pd.Series) -> dict:
    """Full model comparison. Returns tables and fitted objects."""
    train = panel[panel.campaign_k.between(2, 5)].reset_index(drop=True)
    test = panel[panel.campaign_k == 6].reset_index(drop=True)
    y = test["target"].values
    res = {"train_rows": len(train), "test_rows": len(test),
           "train_rate": train.target.mean(), "test_rate": y.mean()}

    feats = FEATURE_SETS["full"]
    tuning = tune(train, feats)
    fac = best_factories(tuning, feats)
    fitted = {n: f().fit(train[feats], train["target"]) for n, f in fac.items()}
    scores = baseline_scores(test, test["ID"].map(rfm_by_id))
    for n, m in fitted.items():
        scores[f"M_{n}"] = m.predict_proba(test[feats])[:, 1]
    comp = evaluate_scores(y, scores)

    # Primary model chosen on TRAINING LOCO CV only
    # Declared rule: prefer the simpler, interpretable LR unless boosting beats it on LOCO PR-AUC by > 0.01
    best = tuning.groupby("model").loco_pr_auc.max()
    primary = ("gradient_boosting" if best["gradient_boosting"] - best["logistic_regression"] > SIMPLICITY_MARGIN
               else "logistic_regression")
    res["primary_model"] = primary
    # paired bootstrap: primary model vs strongest simple baseline (by LOCO the rule has no params,
    # so compare to both rules)
    res["diff_vs_B1"] = bootstrap_auc_diff(y, scores[f"M_{primary}"], scores["B1_prior_acceptances"])
    res["diff_vs_B2"] = bootstrap_auc_diff(y, scores[f"M_{primary}"], scores["B2_RFM_score"])
    res["diff_gb_vs_lr"] = bootstrap_auc_diff(y, scores["M_gradient_boosting"], scores["M_logistic_regression"])

    # Leakage-sensitivity variants (logistic regression, same C)
    C = tuning[tuning.model == "logistic_regression"].sort_values("loco_pr_auc").iloc[-1].C
    var_rows = []
    for name, fs in FEATURE_SETS.items():
        m = make_lr(fs, C).fit(train[fs], train["target"])
        s = m.predict_proba(test[fs])[:, 1]
        var_rows.append({"feature_set": name, "n_features": len(fs), **ranking_metrics(y, s)})
    res["leakage_variants"] = pd.DataFrame(var_rows)

    # In-campaign 5-fold CV on C6 only (reference: optimistic, same-campaign)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    oof = cross_val_predict(make_lr(feats, C), test[feats], y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    res["in_campaign_cv"] = ranking_metrics(y, oof)

    pm = fitted[primary]
    p_test = scores[f"M_{primary}"]
    res.update({"tuning": tuning, "comparison": comp, "models": fitted, "scores": scores, "y_test": y,
                "test_ids": test["ID"].values, "p_test": p_test,
                "calibration_raw": calibration_table(y, p_test),
                "threshold_curve": threshold_curve(y, p_test, 3.0, 11.0),
                "coefficients": logistic_coefficients(fitted["logistic_regression"], feats),
                "perm_importance": perm_importance(pm, test[feats], y),
                "features": feats, "C": C})
    return res


def fit_forward_model(panel: pd.DataFrame, scoring: pd.DataFrame, factory, features, pilot_rate: float):
    """Refit on ALL observed campaigns (C2..C6) and score the next, unobserved campaign.
    Probabilities are prior-shifted to the pilot's base rate (assumption: next campaign behaves
    like the pilot)."""
    m = factory().fit(panel[features], panel["target"])
    raw = m.predict_proba(scoring[features])[:, 1]
    return m, raw, prior_shift(raw, pilot_rate)
