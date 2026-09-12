"""Consistent, restrained chart style + every project figure. Each function answers one question
(stated in the title) and is saved to outputs/figures/."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from .config import FIG, BREAK_EVEN_P

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = "Blues"
DIV = "RdBu_r"
SRC_IFOOD = "Source: iFood CRM case dataset (n=2,019 de-duplicated customers)"
SRC_HILL = "Source: Hillstrom MineThatData e-mail experiment (n=64,000, randomised)"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 9, "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 150,
        "axes.axisbelow": True, "lines.linewidth": 2,
    })


style()


def _finish(fig, name, source=None):
    if source:
        fig.text(0.01, 0.005, source, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if source else 0, 1, 1))
    if not name.startswith("_"):  # names starting with "_" are display-only (dashboard)
        fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    return fig


def pct_axis(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(mtick.PercentFormatter(1, decimals=decimals))


# ------------------------------------------------------------------ campaigns
def campaign_acceptance(cs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(cs))
    ax.bar(x, cs.acceptance_rate, color=[BLUE] * 5 + [ORANGE], width=0.6)
    ax.errorbar(x, cs.acceptance_rate, yerr=[cs.acceptance_rate - cs.ci_low, cs.ci_high - cs.acceptance_rate],
                fmt="none", ecolor=INK2, capsize=3, lw=1)
    for i, r in cs.iterrows():
        ax.text(i, r.ci_high + 0.004, f"{r.acceptance_rate:.1%}\n({r.acceptors})", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, cs.campaign)
    pct_axis(ax)
    ax.set_ylabel("Acceptance rate (share of customers)")
    ax.set_title("Which campaigns worked? Acceptance rate per campaign (95% Wilson CI)")
    ax.set_ylim(0, cs.ci_high.max() * 1.3)
    return _finish(fig, "fig01_campaign_acceptance", SRC_IFOOD)


def heatmap(mat: pd.DataFrame, title, name, fmt="{:.0%}", cmap=SEQ, center=None, xlabel="", ylabel="",
            source=SRC_IFOOD, figsize=(7.5, 5), cbar_label=""):
    fig, ax = plt.subplots(figsize=figsize)
    vals = mat.values.astype(float)
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=min(np.nanmin(vals), center - 1e-3), vcenter=center, vmax=max(np.nanmax(vals), center + 1e-3))
        im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    else:
        im = ax.imshow(vals, cmap=cmap, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=0)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    norm = im.norm
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            lum = norm(v)
            dark = (lum > 0.65) if center is None else abs(lum - 0.5) > 0.32
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(cbar_label, color=INK2); cb.outline.set_visible(False)
    return _finish(fig, name, source)


def rate_bars(df, cat, rate, lo, hi, title, name, xlabel, color=BLUE, ref=None, ref_label=None, source=SRC_IFOOD,
              figsize=(7, 4), annotate_n="n"):
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(df))
    ax.barh(y, df[rate], color=color, height=0.6)
    ax.errorbar(df[rate], y, xerr=[df[rate] - df[lo], df[hi] - df[rate]], fmt="none", ecolor=INK2, capsize=3, lw=1)
    ax.set_yticks(y, df[cat].astype(str)); ax.invert_yaxis()
    for i, r in enumerate(df.itertuples()):
        ax.text(getattr(r, hi) + 0.005, i, f"{getattr(r, rate):.1%}" + (f"  n={getattr(r, annotate_n):,}" if annotate_n else ""),
                va="center", fontsize=8, color=INK2)
    if ref is not None:
        ax.axvline(ref, color=INK2, ls="--", lw=1)
        ax.text(ref, len(df) - 0.4, f" {ref_label}", color=INK2, fontsize=8, va="bottom")
    pct_axis(ax, "x"); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.set_xlim(0, min(1.0, df[hi].max() * 1.35))
    return _finish(fig, name, source)


# ------------------------------------------------------------------ value
def lorenz_curve(lz: pd.DataFrame, gini: float, top20: float):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(lz.share_customers, lz.share_revenue, color=BLUE, label="Observed (customers ranked by spend)")
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, label="Perfect equality")
    ax.fill_between(lz.share_customers, lz.share_revenue, lz.share_customers, color=BLUE, alpha=0.08)
    ax.scatter([0.2], [top20], color=ORANGE, zorder=3, s=40)
    ax.annotate(f"Top 20% of customers\n= {top20:.0%} of 2-yr spend", (0.2, top20), (0.35, top20 - 0.2),
                arrowprops=dict(arrowstyle="-", color=INK2), fontsize=9, color=INK)
    pct_axis(ax, "x"); pct_axis(ax, "y")
    ax.set_xlabel("Cumulative share of customers (highest spend first)")
    ax.set_ylabel("Cumulative share of 2-year spend")
    ax.set_title(f"How concentrated is customer value? (Gini = {gini:.2f})")
    ax.legend(loc="lower right")
    return _finish(fig, "fig04_value_concentration", SRC_IFOOD)


def cohort_chart(co: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(co.enrol_quarter, co.avg_spend_per_month, color=BLUE, width=0.6)
    axes[0].set_ylabel("Avg spend per tenure month (MU)"); axes[0].set_title("Do newer cohorts spend less per month?")
    axes[1].bar(co.enrol_quarter, co.pilot_response, color=ORANGE, width=0.6)
    pct_axis(axes[1]); axes[1].set_ylabel("Pilot (C6) response rate"); axes[1].set_title("Pilot response by enrolment cohort")
    for a in axes:
        a.tick_params(axis="x", rotation=45); a.set_xlabel("Enrolment quarter")
    return _finish(fig, "fig22_cohort_value", SRC_IFOOD + ". Acquisition channel is not recorded.")


# ------------------------------------------------------------------ segmentation
def kmeans_selection(ev: pd.DataFrame, k_star: int):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(ev.k, ev.inertia, marker="o", color=BLUE); axes[0].set_title("Elbow: within-cluster SS")
    axes[0].set_ylabel("Inertia (standardised units)")
    axes[1].plot(ev.k, ev.silhouette, marker="o", color=BLUE); axes[1].set_title("Silhouette (higher = better separated)")
    axes[1].set_ylabel("Mean silhouette")
    axes[2].plot(ev.k, ev.stability_ari_mean, marker="o", color=BLUE, label="mean")
    axes[2].plot(ev.k, ev.stability_ari_min, marker="o", color=ORANGE, label="min")
    axes[2].axhline(0.8, color=INK2, ls="--", lw=1); axes[2].set_title("Bootstrap stability (ARI, 20 resamples)")
    axes[2].set_ylabel("Adjusted Rand index"); axes[2].legend()
    for a in axes:
        a.set_xlabel("Number of clusters k"); a.axvline(k_star, color=GREEN, lw=1, alpha=0.6)
    fig.suptitle(f"How many customer clusters are justified? Selected k = {k_star} (green line)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    return _finish(fig, "fig07_kmeans_selection", SRC_IFOOD)


def fingerprint(prof: pd.DataFrame, cols: dict, title, name):
    sub = prof[list(cols)].rename(columns=cols)
    z = (sub - sub.mean()) / sub.std()
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(sub) + 1.8))
    span = np.nanmax(np.abs(z.values))
    im = ax.imshow(z.values, cmap=DIV, vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=30, ha="right"); ax.set_yticks(range(z.shape[0]), z.index)
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = sub.values[i, j]
            txt = f"{v:.0%}" if abs(v) <= 1 and sub.columns[j].endswith("%") else f"{v:,.0f}" if abs(v) >= 10 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if abs(z.values[i, j]) > 0.65 * span else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("z-score across segments", color=INK2); cb.outline.set_visible(False)
    ax.set_title(title)
    return _finish(fig, name, SRC_IFOOD + ". Cell text = actual value; colour = relative position.")


# ------------------------------------------------------------------ channels
def channel_volume_value(ch: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(ch.channel, ch.purchase_share, color=BLUE, width=0.55); pct_axis(axes[0])
    axes[0].set_title("Volume: share of all purchases"); axes[0].set_ylabel("Share of purchases")
    axes[1].bar(ch.channel, ch.avg_spend_users, color=AQUA, width=0.55)
    axes[1].set_title("Value: avg 2-yr spend of channel users"); axes[1].set_ylabel("MU per customer")
    d = ch.dropna(subset=["dominant_pilot_response"])
    axes[2].bar(d.channel, d.dominant_pilot_response, color=ORANGE, width=0.55)
    axes[2].errorbar(d.channel, d.dominant_pilot_response, yerr=[d.dominant_pilot_response - d.dom_ci_low,
                     d.dom_ci_high - d.dominant_pilot_response], fmt="none", ecolor=INK2, capsize=3, lw=1)
    for i, r in enumerate(d.itertuples()):
        axes[2].text(i, r.dom_ci_high + 0.01, f"n={r.dominant_customers}", ha="center", fontsize=8, color=INK2)
    pct_axis(axes[2]); axes[2].set_title("Pilot response: channel-dominant buyers"); axes[2].set_ylabel("Response rate")
    fig.suptitle("Purchase channels: the highest-volume channel is not the highest-value one", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    return _finish(fig, "fig09_channel_volume_value", SRC_IFOOD + ". Channel = where customers buy; campaign delivery channel is not recorded.")


# ------------------------------------------------------------------ model
def leakage_recency(auc_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(auc_df)); w = 0.38
    ax.bar(x - w / 2, auc_df.auc_recency, w, color=ORANGE, label="Recency (fewer days = higher score)")
    ax.bar(x + w / 2, auc_df.auc_spend, w, color=BLUE, label="2-yr spend")
    ax.axhline(0.5, color=INK2, ls="--", lw=1)
    ax.set_xticks(x, auc_df.campaign); ax.set_ylim(0.3, 1); ax.set_ylabel("ROC-AUC of single feature")
    ax.set_title("Leakage check: Recency predicts ONLY the pilot (C6), not C1-C5")
    ax.legend(loc="upper left")
    return _finish(fig, "fig10_recency_leakage_check", SRC_IFOOD + ". AUC 0.5 = no information.")


def model_comparison(comp: pd.DataFrame, base_rate: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = comp.approach.str.replace("_", " ")
    for ax, m, ci, ref, lab in [(axes[0], "roc_auc", "roc_auc_ci", 0.5, "random = 0.5"),
                                (axes[1], "pr_auc", "pr_auc_ci", base_rate, f"base rate = {base_rate:.3f}")]:
        y = np.arange(len(comp))
        cols = [INK2 if a.startswith("B0") else (BLUE if a.startswith("B") else ORANGE) for a in comp.approach]
        ax.barh(y, comp[m], color=cols, height=0.6)
        lo = np.array([c[0] for c in comp[ci]]); hi = np.array([c[1] for c in comp[ci]])
        ax.errorbar(comp[m], y, xerr=[comp[m] - lo, hi - comp[m]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.set_yticks(y, names); ax.invert_yaxis(); ax.axvline(ref, color=INK2, ls="--", lw=1)
        ax.set_xlabel(m.upper().replace("_", "-") + f" (test campaign C6; {lab})")
        for i, v in enumerate(comp[m]):
            ax.text(hi[i] + 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    axes[0].set_title("Does the model beat simple rules on a later campaign?")
    axes[1].set_title("Precision-recall (imbalance-aware)")
    return _finish(fig, "fig11_model_vs_baselines", SRC_IFOOD + ". Trained on C2-C5, tested on C6. Bars: 95% bootstrap CI.")


def gains_curves(y, scores: dict):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"B1_prior_acceptances": BLUE, "B2_RFM_score": AQUA, "M_logistic_regression": ORANGE,
              "M_gradient_boosting": VIOLET}
    y = np.asarray(y)
    for name, s in scores.items():
        if name.startswith("B0"):
            continue
        o = np.argsort(-np.asarray(s, float), kind="mergesort")
        g = np.r_[0, np.cumsum(y[o]) / y.sum()]
        ax.plot(np.linspace(0, 1, len(g)), g, color=colors.get(name, INK2), label=name.replace("_", " "), lw=1.8)
    ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, label="random")
    pct_axis(ax, "x"); pct_axis(ax, "y")
    ax.set_xlabel("Share of customers contacted (ranked by score)"); ax.set_ylabel("Share of pilot responders captured")
    ax.set_title("Cumulative gains on the pilot campaign (C6)"); ax.legend(loc="lower right")
    return _finish(fig, "fig12_cumulative_gains", SRC_IFOOD)


def calibration_plot(cal_raw: pd.DataFrame, cal_shift: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(cal_raw.mean_predicted, cal_raw.observed_rate, marker="o", color=BLUE, label="as trained (C2-C5 base rate)")
    ax.plot(cal_shift.mean_predicted, cal_shift.observed_rate, marker="o", color=ORANGE, label="prior-shifted to C6 rate*")
    m = max(cal_raw.observed_rate.max(), cal_shift.mean_predicted.max()) * 1.05
    ax.plot([0, m], [0, m], color=INK2, ls="--", lw=1)
    ax.axvline(BREAK_EVEN_P, color=RED, lw=1, alpha=0.6); ax.text(BREAK_EVEN_P, 0.02, " break-even 0.273", color=RED, fontsize=8)
    pct_axis(ax, "x"); pct_axis(ax, "y"); ax.set_xlabel("Mean predicted probability (decile)"); ax.set_ylabel("Observed response rate")
    ax.set_title("Calibration on the pilot: ranking transfers, level does not"); ax.legend(loc="upper left", fontsize=8)
    return _finish(fig, "fig19_calibration", "*uses the observed C6 base rate: illustrative only, not available ex-ante.")


def importance_plot(pi: pd.DataFrame, top=12):
    d = pi.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(d.feature, d.pr_auc_drop_mean, xerr=d.pr_auc_drop_std, color=BLUE, height=0.6, ecolor=INK2)
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled (test C6)")
    ax.set_title("What drives predicted response? (permutation importance)")
    return _finish(fig, "fig20_permutation_importance", SRC_IFOOD)


# ------------------------------------------------------------------ allocation
def budget_plot(bc: pd.DataFrame, hindsight_best=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"Model (temporal LR)": ORANGE, "Prior-acceptance rule": BLUE, "RFM score": AQUA,
              "Historical value (spend)": VIOLET, "Random / mass": INK2}
    for s, d in bc.groupby("strategy", sort=False):
        ax.plot(d.budget_mu, d.profit_mu, label=s, color=colors.get(s, INK2), lw=2 if "Model" in s else 1.6,
                ls="--" if s.startswith("Random") else "-")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Campaign budget (MU; 3 MU per contact)"); ax.set_ylabel("Back-tested profit on pilot (MU)")
    ax.set_title("How should a fixed budget be spent? Profit by budget and targeting strategy")
    ax.legend(loc="lower left")
    return _finish(fig, "fig13_budget_allocation_curve", SRC_IFOOD + ". Back-test on actual C6 outcomes; revenue 11 MU/response is campaign revenue, not margin.")


def marginal_plot(mr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = (mr.from_share + mr.to_share) / 2
    cols = [GREEN if v > 0 else RED for v in mr.marginal_profit_mu]
    ax.bar(x, mr.marginal_profit_mu, width=mr.to_share - mr.from_share - 0.004, color=cols)
    pct_axis(ax, "x"); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("Customers contacted, in model-rank order (5% slices)"); ax.set_ylabel("Profit added by slice (MU)")
    ax.set_title("Diminishing returns: only the first slices pay back the 3 MU contact cost")
    return _finish(fig, "fig14_marginal_returns", SRC_IFOOD)


def tradeoff_plot(pm: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (m, d) in enumerate(pm.groupby("method", sort=False)):
        ax.plot(d.responses, d.avg_hist_spend_responders, marker="o", color=CAT[i % 8], label=m)
        for r in d.itertuples():
            ax.text(r.responses, r.avg_hist_spend_responders, f" {int(r.contact_share*100)}%", fontsize=7.5, color=INK2)
    ax.set_xlabel("Pilot responders captured (volume)"); ax.set_ylabel("Avg historical spend of captured responders (MU)")
    ax.set_title("Response volume vs customer value, by prioritisation method")
    ax.legend(fontsize=8, loc="best")
    return _finish(fig, "fig15_response_vs_value_tradeoff", SRC_IFOOD + ". Labels = share of base contacted.")


# ------------------------------------------------------------------ hillstrom
def ate_plot(ate: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, o, unit in zip(axes, ["visit", "conversion", "spend"], ["pp", "pp", "$"]):
        d = ate[ate.outcome == o]
        y = np.arange(len(d))
        sc = 100 if unit == "pp" else 1
        ax.errorbar(d.effect * sc, y, xerr=[(d.effect - d.ci_low) * sc, (d.ci_high - d.effect) * sc], fmt="o",
                    color=BLUE, ecolor=BLUE, capsize=3)
        ax.axvline(0, color=INK2, ls="--", lw=1)
        ax.set_yticks(y, d.arm); ax.invert_yaxis()
        ax.set_xlabel(f"Effect ({'percentage points' if unit == 'pp' else '$ per customer'})")
        ax.set_title(o.capitalize())
    fig.suptitle("Incremental effect of each e-mail vs no e-mail (randomised, 95% CI)", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold")
    return _finish(fig, "fig16_hillstrom_treatment_effects", SRC_HILL)


def subgroup_forest(sub: pd.DataFrame, dim: str, outcome: str):
    d = sub[(sub.dimension == dim) & (sub.outcome == outcome)]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(d) + 1.5))
    sc = 100 if outcome != "spend" else 1
    labels, y = [], 0
    for arm, col in [("Mens", BLUE), ("Womens", ORANGE)]:
        dd = d[d.arm == arm]
        for r in dd.itertuples():
            ax.errorbar(r.effect * sc, y, xerr=[[(r.effect - r.ci_low) * sc], [(r.ci_high - r.effect) * sc]], fmt="o",
                        color=col, ecolor=col, capsize=3, mfc=col if r.significant_bh_5pct else SURF)
            labels.append(f"{arm} e-mail | {r.level}"); y += 1
    ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis(); ax.axvline(0, color=INK2, ls="--", lw=1)
    ax.set_xlabel(f"Incremental {outcome} ({'pp' if sc == 100 else '$ per customer'})")
    ax.set_title(f"Which e-mail works for whom? {outcome} effect by {dim.replace('_', ' ')}")
    return _finish(fig, f"fig17_hillstrom_{dim}_{outcome}", SRC_HILL + ". Filled = significant after BH correction.")


def policy_plot(pol: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    d = pol[pol.policy != "No e-mail"]
    y = np.arange(len(d))
    ax.barh(y, d.incremental_spend_per_customer, color=BLUE, height=0.55)
    ax.errorbar(d.incremental_spend_per_customer, y, xerr=[d.incremental_spend_per_customer - d.inc_spend_ci_low,
                d.inc_spend_ci_high - d.incremental_spend_per_customer], fmt="none", ecolor=INK2, capsize=3)
    ax.set_yticks(y, d.policy); ax.invert_yaxis()
    for i, r in enumerate(d.itertuples()):
        ax.text(r.inc_spend_ci_high + 0.02, i, f"${r.incremental_spend_per_customer:.2f}  ({r.emails_per_customer:.0%} e-mailed)",
                va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Incremental spend per customer vs no e-mail ($, held-out half, IPW)")
    ax.set_title("Does personalised campaign assignment beat 'send the best e-mail to all'?")
    ax.set_xlim(0, d.inc_spend_ci_high.max() * 1.45)
    return _finish(fig, "fig18_hillstrom_policy_value", SRC_HILL)


def action_matrix_plot(mat: pd.DataFrame):
    return heatmap(mat, "Customer action matrix: value tier x targeting priority (next campaign)",
                   "fig21_action_matrix", fmt="{:.0f}", xlabel="Targeting priority", ylabel="RFM value tier",
                   cbar_label="customers", figsize=(8, 3.8))
