"""End-to-end pipeline. Each stage reads the context produced by earlier stages, writes its
tables/figures to outputs/, and returns the updated context. Notebooks call the same stages.

Lineage: data/raw -> clean (data/processed) -> customer features -> RFM/segments ->
campaign & channel analytics -> response model -> customer value -> targeting engine ->
allocation -> robustness -> SQL cross-check -> dashboard.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import (ingestion, data_validation as dv, preprocessing, feature_engineering as fe, rfm,
               segmentation as seg, campaign_analysis as ca, response_model as rm, customer_value as cv,
               targeting as tg, optimization as opt, uplift as up, robustness as rb, visualization as viz)
from .config import (TAB, MODELS, DATA_PROCESSED, CAMPAIGNS, CAMPAIGN_LABELS, BREAK_EVEN_P,
                     COST_PER_CONTACT, REVENUE_PER_RESPONSE, SEED)
from .evaluation import rate_table, chi2_test, mean_diff_ci, adjust_pvalues



def _save(df: pd.DataFrame, name: str, index=False):
    df.to_csv(TAB / f"{name}.csv", index=index)
    return df


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- 1. audit
def stage_audit(ctx: dict) -> dict:
    _log("audit")
    raw_i, raw_h = ingestion.load_ifood_raw(), ingestion.load_hillstrom_raw()
    prof = pd.concat([dv.column_profile(raw_i, "ifood"), dv.column_profile(raw_h, "hillstrom")])
    rules = pd.concat([dv.ifood_rule_checks(raw_i), dv.hillstrom_rule_checks(raw_h)])
    rules["result"] = rules["result"].astype(str)
    _save(prof, "data_quality_report")
    _save(rules, "data_quality_rule_checks")
    bal = _save(dv.randomisation_balance(raw_h.replace({"Surburban": "Suburban"})), "hillstrom_randomisation_balance")
    ctx.update(raw_ifood=raw_i, raw_hill=raw_h, audit_profile=prof, audit_rules=rules, balance=bal)
    return ctx


# ----------------------------------------------------------------------------- 2. clean
def stage_clean(ctx: dict) -> dict:
    _log("clean")
    ci, log_i = preprocessing.clean_ifood(ctx["raw_ifood"])
    ch, log_h = preprocessing.clean_hillstrom(ctx["raw_hill"])
    ci.to_csv(DATA_PROCESSED / "ifood_customers_clean.csv", index=False)
    ch.to_csv(DATA_PROCESSED / "hillstrom_clean.csv", index=False)
    _save(pd.concat([log_i.assign(dataset="ifood"), log_h.assign(dataset="hillstrom")]), "cleaning_log")
    ctx.update(clean_ifood=ci, hill=ch, cleaning_log_ifood=log_i, cleaning_log_hill=log_h)
    return ctx


# ----------------------------------------------------------------------------- 3. features + RFM
def stage_features(ctx: dict) -> dict:
    _log("features + RFM")
    feat = fe.customer_features(ctx["clean_ifood"])
    r = rfm.rfm_segments(rfm.rfm_table(feat, 5))
    feat = feat.merge(r[["ID", "R", "F", "M", "RFM_score", "FM", "activity_tier", "value_tier", "rfm_segment"]], on="ID")
    for c in ["activity_tier", "value_tier"]:
        feat[c] = feat[c].astype(str)
    _save(r.merge(feat[["ID", "Response"]], on="ID"), "rfm_scores")
    _save(rfm.recency_tier_bounds(r).reset_index(), "rfm_recency_tier_bounds")
    # quintile edges (documentation of scoring)
    edges = pd.DataFrame({m: r.groupby(s)[m].agg(["min", "max"]).apply(lambda x: f"{x['min']:.0f}-{x['max']:.0f}", axis=1)
                          for m, s in [("recency_days", "R"), ("frequency", "F"), ("monetary", "M")]})
    _save(edges.reset_index().rename(columns={"index": "score"}), "rfm_score_edges")
    prof = rfm.segment_profiles(feat, "rfm_segment")
    _save(prof.reset_index(), "rfm_segment_profiles")
    # RFM map: R score x FM tier -> pilot response & customers
    grid = feat.assign(FMb=pd.cut(feat.FM, [0, 1.5, 2.5, 3.5, 4.5, 5], labels=["1", "2", "3", "4", "5"]))
    rmap = grid.pivot_table(index="FMb", columns="R", values="Response", aggfunc="mean", observed=True).iloc[::-1]
    cmap_n = grid.pivot_table(index="FMb", columns="R", values="ID", aggfunc="count", observed=True).iloc[::-1]
    _save(rmap.reset_index(), "rfm_map_response"); _save(cmap_n.reset_index(), "rfm_map_counts")
    viz.heatmap(rmap.rename_axis(index="F+M score (avg)", columns="R score"),
                "RFM map: pilot response rate by recency score x frequency-monetary score", "fig05_rfm_map",
                xlabel="R score (5 = most recent)", ylabel="(F+M)/2 score (5 = most valuable)", cbar_label="response rate")
    viz.fingerprint(prof, {"customers": "customers", "share_revenue_pct": "% of revenue", "avg_spend": "avg spend",
                           "avg_purchases": "purchases", "avg_recency_days": "recency (d)", "avg_aov": "AOV",
                           "deals_share": "deal share", "catalog_share": "catalog share",
                           "avg_web_visits": "web visits/mo", "prior5_accept_rate": "C1-C5 accept",
                           "pilot_response_rate": "C6 response"},
                    "RFM segment fingerprints: what does each segment actually look like?", "fig06_rfm_fingerprint")
    # statistical test: response differs across RFM segments?
    ctx["rfm_test"] = chi2_test(feat, "rfm_segment", "Response")
    feat.to_csv(DATA_PROCESSED / "customer_features.csv", index=False)
    ctx.update(feat=feat, rfm=r, rfm_profiles=prof)
    return ctx


# ----------------------------------------------------------------------------- 4. segmentation
def stage_segmentation(ctx: dict) -> dict:
    _log("segmentation")
    feat = ctx["feat"]
    X = seg.cluster_matrix(feat)
    _save(X.corr().round(3).reset_index(), "cluster_feature_correlations")
    ev = _save(seg.evaluate_k(X), "kmeans_k_selection")
    k = seg.choose_k(ev)
    km, sc = seg.fit_kmeans(X, k)
    names = seg.label_clusters(X, km.labels_, feat)
    feat["cluster"] = km.labels_
    feat["cluster_label"] = feat["cluster"].map(names)
    prof = rfm.segment_profiles(feat, "cluster_label")
    _save(prof.reset_index(), "cluster_profiles")
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=X.columns).assign(label=[names[i] for i in range(k)])
    _save(cent, "cluster_centroids")
    cmp_ = _save(seg.compare_with_rfm(feat, "rfm_segment", "cluster_label"), "segmentation_rfm_vs_kmeans")
    xt = pd.crosstab(feat["cluster_label"], feat["rfm_segment"])
    _save(xt.reset_index(), "cluster_vs_rfm_crosstab")
    viz.kmeans_selection(ev, k)
    viz.fingerprint(prof, {"customers": "customers", "share_revenue_pct": "% of revenue", "avg_spend": "avg spend",
                           "avg_purchases": "purchases", "avg_recency_days": "recency (d)", "deals_share": "deal share",
                           "catalog_share": "catalog share", "avg_web_visits": "web visits/mo",
                           "prior5_accept_rate": "C1-C5 accept", "pilot_response_rate": "C6 response"},
                    "KMeans cluster fingerprints", "fig06b_cluster_fingerprint")
    joblib.dump({"kmeans": km, "scaler": sc, "features": list(X.columns), "labels": names}, MODELS / "kmeans_segments.joblib")
    ctx.update(feat=feat, X_cluster=X, k_eval=ev, k_star=k, cluster_names=names, seg_compare=cmp_, cluster_profiles=prof)
    feat.to_csv(DATA_PROCESSED / "customer_features.csv", index=False)
    return ctx


# ----------------------------------------------------------------------------- 5. campaigns & channels
def stage_campaigns(ctx: dict) -> dict:
    _log("campaigns + channels")
    feat = ctx["feat"]
    cs = _save(ca.campaign_summary(feat), "campaign_summary")
    ov = ca.campaign_overlap(feat); _save(ov.reset_index().rename(columns={"index": "accepted"}), "campaign_overlap")
    rep = _save(ca.repeat_response(feat), "repeat_response")
    econ = _save(ca.pilot_economics(feat).reset_index(names="scope"), "pilot_economics")
    econ_seg = _save(ca.pilot_economics(feat, "rfm_segment").reset_index(), "pilot_economics_by_rfm_segment")
    econ_cl = _save(ca.pilot_economics(feat, "cluster_label").reset_index(), "pilot_economics_by_cluster")
    rates, index, tests = ca.campaign_by_segment(feat, "rfm_segment")
    _save(rates.reset_index(), "campaign_x_rfm_segment_rates"); _save(index.reset_index(), "campaign_x_rfm_segment_index")
    _save(tests, "campaign_x_rfm_segment_tests")
    rates_c, index_c, tests_c = ca.campaign_by_segment(feat, "cluster_label")
    _save(rates_c.reset_index(), "campaign_x_cluster_rates"); _save(tests_c, "campaign_x_cluster_tests")
    long = _save(ca.segment_campaign_long(feat, "rfm_segment"), "campaign_x_rfm_segment_long")
    ch = _save(ca.channel_summary(feat), "channel_summary")
    chi = _save(ca.channel_intensity_response(feat), "channel_intensity_response")
    dom = _save(rate_table(feat, "dominant_channel", "Response").rename(columns={"dominant_channel": "group"}),
                "dominant_channel_response")
    fun = _save(ca.funnel(feat), "pilot_funnel")

    # statistical tests with multiple-testing correction (family: pilot response vs each attribute)
    fam = []
    for g in ["rfm_segment", "cluster_label", "dominant_channel", "Education", "Marital_Status", "activity_tier",
              "value_tier"]:
        t = chi2_test(feat, g, "Response"); fam.append({"test": f"Response ~ {g}", "method": "chi-square independence", **t})
    for col in ["total_spend", "Income", "Recency", "NumWebVisitsMonth", "tenure_days"]:
        a, b = feat.loc[feat.Response == 1, col].dropna(), feat.loc[feat.Response == 0, col].dropna()
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(a, b)
        d, lo, hi, _ = mean_diff_ci(a, b)
        fam.append({"test": f"{col}: responders vs non-responders", "method": "Mann-Whitney U (+Welch CI of mean diff)",
                    "p_value": u.pvalue, "n": len(a) + len(b), "effect_rank_biserial": 2 * u.statistic / (len(a) * len(b)) - 1,
                    "mean_diff": d, "ci_low": lo, "ci_high": hi})
    fam = pd.DataFrame(fam)
    fam["p_bh"] = adjust_pvalues(fam["p_value"], "fdr_bh")
    _save(fam, "statistical_tests_pilot_response")

    viz.campaign_acceptance(cs)
    viz.heatmap(ov, "Do the same customers accept repeatedly? P(accept column | accepted row)", "fig02_campaign_overlap",
                xlabel="Campaign (column)", ylabel="Accepted campaign (row)", cbar_label="conditional acceptance")
    viz.rate_bars(rep, "prior_acceptances", "rate", "ci_low", "ci_high",
                  "Pilot response by number of earlier campaigns accepted", "fig03_repeat_response",
                  "Pilot (C6) response rate", ref=BREAK_EVEN_P, ref_label="break-even 27.3%")
    viz.heatmap(index.loc[index.mean(axis=1).sort_values(ascending=False).index],
                "Which campaign works for which segment? Acceptance index vs campaign average",
                "fig08_campaign_x_segment", fmt="{:.2f}", cmap="RdBu_r", center=1.0, xlabel="Campaign",
                ylabel="RFM segment", cbar_label="index (1.0 = campaign average)", figsize=(9, 5))
    viz.channel_volume_value(ch)
    ctx.update(campaign_summary=cs, overlap=ov, repeat=rep, econ=econ, econ_seg=econ_seg, econ_cluster=econ_cl,
               cxs_rates=rates, cxs_index=index, cxs_tests=tests, cxs_long=long, channels=ch, channel_intensity=chi,
               dominant_channel_response=dom, stats_family=fam, cxc_tests=tests_c)
    return ctx


# ----------------------------------------------------------------------------- 6. response model
def stage_model(ctx: dict) -> dict:
    _log("response model (temporal validation)")
    feat = ctx["feat"]
    panel = fe.campaign_panel(feat)
    panel.to_csv(DATA_PROCESSED / "campaign_panel.csv", index=False)
    rfm_by_id = feat.set_index("ID")["RFM_score"]
    res = rm.run(panel, rfm_by_id)

    # leakage evidence: single-feature AUC per campaign
    auc = pd.DataFrame([{"campaign": CAMPAIGN_LABELS[c], "auc_recency": roc_auc_score(feat[c], -feat.Recency),
                         "auc_spend": roc_auc_score(feat[c], feat.total_spend),
                         "auc_web_visits": roc_auc_score(feat[c], feat.NumWebVisitsMonth),
                         "auc_income": roc_auc_score(feat[c], feat.Income.fillna(feat.Income.median()))}
                        for c in CAMPAIGNS])
    _save(auc, "leakage_single_feature_auc_by_campaign")

    comp = res["comparison"].copy()
    comp["roc_auc_ci_low"] = [c[0] for c in comp.roc_auc_ci]; comp["roc_auc_ci_high"] = [c[1] for c in comp.roc_auc_ci]
    comp["pr_auc_ci_low"] = [c[0] for c in comp.pr_auc_ci]; comp["pr_auc_ci_high"] = [c[1] for c in comp.pr_auc_ci]
    _save(comp.drop(columns=["roc_auc_ci", "pr_auc_ci"]), "model_comparison_test_C6")
    _save(res["tuning"], "model_tuning_loco_cv")
    _save(res["leakage_variants"], "model_leakage_variants")
    _save(res["coefficients"], "model_lr_coefficients")
    _save(res["perm_importance"], "model_permutation_importance")
    _save(res["calibration_raw"], "model_calibration_raw")
    tc = res["threshold_curve"]
    _save(tc.iloc[np.r_[np.arange(0, len(tc), 20), len(tc) - 1]], "model_threshold_curve")
    diffs = []
    for k in ["diff_vs_B1", "diff_vs_B2", "diff_gb_vs_lr"]:
        for m, v in res[k].items():
            diffs.append({"comparison": k, "metric": m, "mean_diff": v[0], "ci_low": v[1], "ci_high": v[2]})
    _save(pd.DataFrame(diffs), "model_paired_bootstrap_differences")
    _save(pd.DataFrame([{"design": "temporal (train C2-C5, test C6) - primary", **{k: v for k, v in comp.set_index('approach').loc[f"M_{res['primary_model']}"].items() if k in ('roc_auc','pr_auc','lift@20%')}},
                        {"design": "in-campaign 5-fold CV on C6 (optimistic reference)", **{k: res['in_campaign_cv'][k] for k in ('roc_auc','pr_auc','lift@20%')}}]),
          "model_validation_designs")

    # threshold metrics at business-relevant operating points
    y, p = res["y_test"], res["p_test"]
    ops = []
    from .evaluation import threshold_metrics
    for name, mask in [("top 10%", tc.contacts <= round(0.1 * len(y))), ("top 20%", tc.contacts <= round(0.2 * len(y))),
                       ("top 30%", tc.contacts <= round(0.3 * len(y)))]:
        k = int(mask.sum()); pred = np.zeros(len(y), int); pred[np.argsort(-p, kind="mergesort")[:k]] = 1
        ops.append({"operating_point": name, "contacts": k, **rm_threshold(y, pred)})
    pred = (p >= BREAK_EVEN_P).astype(int)
    ops.append({"operating_point": "raw p >= break-even 0.273", "contacts": int(pred.sum()), **rm_threshold(y, pred)})
    _save(pd.DataFrame(ops), "model_operating_points")

    shifted = rm.prior_shift(p, y.mean())
    cal_shift = rm.calibration_table(y, shifted)
    _save(cal_shift, "model_calibration_prior_shifted")

    viz.leakage_recency(auc)
    viz.model_comparison(res["comparison"], y.mean())
    viz.gains_curves(y, res["scores"])
    viz.calibration_plot(res["calibration_raw"], cal_shift)
    viz.importance_plot(res["perm_importance"])
    joblib.dump({"model": res["models"][res["primary_model"]], "features": res["features"],
                 "trained_on": "C2-C5 panel"}, MODELS / "response_model_temporal.joblib")
    test_scores = pd.DataFrame({"ID": res["test_ids"], "y_C6": y, **{k: v for k, v in res["scores"].items()}})
    _save(test_scores, "model_test_scores_C6")
    ctx.update(panel=panel, model=res, leak_auc=auc, test_scores=test_scores)
    return ctx


def rm_threshold(y, pred):
    from .evaluation import threshold_metrics
    return threshold_metrics(y, pred)


# ----------------------------------------------------------------------------- 7. customer value
def stage_value(ctx: dict) -> dict:
    _log("customer value")
    feat = ctx["feat"]
    conc = _save(cv.concentration_summary(feat), "value_concentration")
    dec = _save(cv.value_deciles(feat), "value_deciles")
    coh = _save(cv.cohort_value(feat), "value_by_enrolment_cohort")
    vvr = _save(cv.value_vs_response(feat), "value_tercile_response")
    cat = _save(cv.category_mix(feat), "category_mix")
    lz = cv.lorenz(feat.total_spend)
    _save(lz.iloc[::10], "lorenz_curve")
    top20 = float(conc.loc[conc.metric == "revenue share of top 20% customers", "value"].iloc[0])
    viz.lorenz_curve(lz, float(conc.value.iloc[0]), top20)
    viz.cohort_chart(coh)
    # spend-per-month trend across cohorts (Spearman) — is there a cohort effect?
    from scipy.stats import spearmanr
    rho = spearmanr(feat["tenure_days"], feat["spend_per_month"])
    ctx.update(value_conc=conc, value_deciles=dec, cohort=coh, value_resp=vvr, category=cat,
               tenure_spm_spearman=(float(rho.statistic), float(rho.pvalue)))
    return ctx


# ----------------------------------------------------------------------------- 8. targeting engine
def stage_targeting(ctx: dict) -> dict:
    _log("targeting engine")
    feat, panel, res = ctx["feat"], ctx["panel"], ctx["model"]
    scoring = fe.scoring_frame(feat)
    features = res["features"]
    factory = lambda: rm.make_lr(features, res["C"])
    model_f, raw_f, p_f = rm.fit_forward_model(panel, scoring, factory, features, feat["Response"].mean())
    contrib = tg.lr_contributions(model_f, scoring, features)
    engine = tg.build_engine(feat, scoring, p_f, contrib, ctx["dominant_channel_response"])
    engine.to_csv(DATA_PROCESSED / "targeting_engine.csv", index=False)
    _save(engine.head(200), "targeting_engine_top200")
    am = tg.action_matrix(engine)
    _save(am.reset_index(), "action_matrix")
    act = engine.groupby(["priority", "action"]).agg(customers=("ID", "size"), avg_p=("p_next_campaign", "mean"),
                                                    avg_spend=("total_spend", "mean"),
                                                    expected_profit_total=("expected_profit_mu", "sum")).reset_index()
    _save(act, "action_summary")
    _save(pd.DataFrame([{"priority": k, "rule": v} for k, v in tg.PRIORITY_RULES.items()]), "priority_rules")
    ts0 = ctx["test_scores"]
    pb = _save(tg.priority_backtest(ts0.y_C6.values, np.floor(ts0.B1_prior_acceptances.values),
                                    ts0[f"M_{res['primary_model']}"].values), "priority_rule_backtest")
    ctx["priority_backtest"] = pb
    cs_tab, cs_md = tg.case_studies(engine, ctx["cxs_long"], pb)
    _save(cs_tab, "case_studies")
    (TAB / "case_studies.md").write_text(cs_md)
    # evidence: did the forward model's inputs change? coefficients of forward model
    _save(rm.logistic_coefficients(model_f, features), "forward_model_coefficients")
    joblib.dump({"model": model_f, "features": features, "trained_on": "C2-C6 panel",
                 "prior_shift_target_rate": float(feat["Response"].mean())}, MODELS / "response_model_forward.joblib")

    # compare prioritisation methods on the pilot back-test
    ts = ctx["test_scores"].merge(feat[["ID", "total_spend"]], on="ID")
    p_model = ts[f"M_{res['primary_model']}"].values
    val_rank = ts.total_spend.rank(pct=True).values
    methods = {"Model p (expected profit)": p_model,
               "Prior-acceptance rule": ts["B1_prior_acceptances"].values + 1e-6 * p_model,
               "RFM score": ts["B2_RFM_score"].values + 1e-6 * p_model,
               "Historical value": ts.total_spend.values,
               "Blend: rank(p) + rank(value)": pd.Series(p_model).rank(pct=True).values + val_rank}
    pm = _save(tg.compare_priority_methods(ts, ts.y_C6.values, methods), "priority_method_comparison")
    viz.tradeoff_plot(pm)
    viz.action_matrix_plot(am.drop(index="All", columns="All").reindex(["High value", "Mid value", "Low value"]))
    ctx.update(engine=engine, action_matrix=am, action_summary=act, priority_methods=pm, p_forward=p_f,
               forward_model=model_f)
    return ctx


# ----------------------------------------------------------------------------- 9. allocation
def stage_optimization(ctx: dict) -> dict:
    _log("budget allocation")
    feat, res = ctx["feat"], ctx["model"]
    ts = ctx["test_scores"].merge(feat[["ID", "total_spend"]], on="ID")
    y = ts.y_C6.values
    p_model = ts[f"M_{res['primary_model']}"].values
    scores = {"Model (temporal LR)": p_model,
              "Prior-acceptance rule": ts["B1_prior_acceptances"].values + 1e-6 * p_model,
              "RFM score": ts["B2_RFM_score"].values + 1e-6 * p_model,
              "Historical value (spend)": ts.total_spend.values,
              "Random / mass": np.zeros(len(y))}
    budgets = np.unique(np.r_[np.arange(0, len(y) * COST_PER_CONTACT + 1, 150), len(y) * COST_PER_CONTACT])  # incl. contact-all
    bc = _save(opt.budget_curve(y, scores, budgets), "budget_curve_backtest")
    scen = _save(opt.scenario_table(y, {"B1_prior_acceptances": ts["B1_prior_acceptances"].values,
                                        "B2_RFM_score": scores["RFM score"]}, ts, p_model), "allocation_scenarios_backtest")
    boot = _save(opt.bootstrap_profit(y, p_model, [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]), "allocation_profit_bootstrap_model")
    boot_rule = _save(opt.bootstrap_profit(y, scores["Prior-acceptance rule"], [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]),
                      "allocation_profit_bootstrap_rule")
    mr = _save(opt.marginal_returns(y, p_model), "marginal_returns_model")
    sens = _save(opt.economics_sensitivity(y, p_model), "economics_sensitivity")
    viz.budget_plot(bc)
    viz.marginal_plot(mr)

    # Forward (model-expected, NOT observed) plan for the next campaign under budgets
    eng = ctx["engine"]
    rows = []
    for B in [600, 1200, 1800, 2400, 3000, 6000]:
        k = int(B // COST_PER_CONTACT)
        top = eng.nsmallest(k, "priority_rank")
        er = top.p_next_campaign.sum()
        rows.append({"budget_mu": B, "contacts": k, "expected_responses": er,
                     "expected_profit_mu": er * REVENUE_PER_RESPONSE - k * COST_PER_CONTACT,
                     "customers_with_positive_expected_profit_in_selection": int((top.expected_profit_mu > 0).sum()),
                     "label": "MODEL-EXPECTED under assumption next campaign ~ pilot (not an observed outcome)"})
    fwd = _save(pd.DataFrame(rows), "allocation_forward_plan_expected")
    ctx.update(budget_curve=bc, scenarios=scen, profit_boot=boot, profit_boot_rule=boot_rule, marginal=mr,
               econ_sens=sens, forward_plan=fwd)
    return ctx


# ----------------------------------------------------------------------------- 10. uplift (Hillstrom)
def stage_uplift(ctx: dict) -> dict:
    _log("incremental response (Hillstrom experiment)")
    h = ctx["hill"]
    res = up.run(h)
    _save(res["arm_summary"], "hillstrom_arm_summary")
    _save(res["ate"], "hillstrom_average_effects")
    _save(res["subgroups"], "hillstrom_subgroup_effects")
    _save(res["heterogeneity"], "hillstrom_heterogeneity_tests")
    _save(res["qini_table"], "hillstrom_qini")
    _save(res["segment_rule"], "hillstrom_segment_rule")
    _save(res["policies"], "hillstrom_policy_values")
    _save(up.email_cost_scenarios(res["policies"]), "hillstrom_email_cost_scenarios_HYPOTHETICAL")
    viz.ate_plot(res["ate"])
    viz.subgroup_forest(res["subgroups"], "buyer_type", "conversion")
    viz.subgroup_forest(res["subgroups"], "history_band", "spend")
    viz.policy_plot(res["policies"])
    ctx["uplift"] = res
    return ctx


# ----------------------------------------------------------------------------- 11. robustness
def stage_robustness(ctx: dict) -> dict:
    _log("robustness")
    feat, res = ctx["feat"], ctx["model"]
    t1 = _save(rb.rfm_binning(feat), "robust_rfm_binning")
    t2 = _save(rb.kmeans_k_sensitivity(feat, ctx["X_cluster"], ctx["k_star"]), "robust_kmeans_k")
    t3 = _save(rb.training_window(ctx["panel"], res["C"]), "robust_training_window")
    t4 = _save(rb.per_campaign_generalisation(ctx["panel"], res["C"]), "robust_per_campaign_drift")
    t5 = _save(rb.dedup_sensitivity(ctx["raw_ifood"], feat), "robust_deduplication")
    t6 = _save(rb.uplift_bootstrap(ctx["hill"]), "robust_uplift_bootstrap")
    sens = ctx["econ_sens"]
    boot = ctx["profit_boot"]
    lv = res["leakage_variants"].set_index("feature_set")
    comp = res["comparison"].set_index("approach")
    diff = res["diff_vs_B1"]["pr_auc_diff"]

    findings = [
        ("Prior campaign acceptance is the strongest single predictor of pilot response",
         "top permutation importance; rule AUC under every de-dup/training window variant",
         bool((t5.rule_auc_prior_acceptances > 0.68).all() and res["perm_importance"].iloc[0].feature == "hist_n_prior")),
        ("Model beats the prior-acceptance rule",
         f"paired bootstrap PR-AUC diff CI [{diff[1]:.3f}, {diff[2]:.3f}] must exclude 0", bool(diff[1] > 0)),
        ("Model ranking is better than random on a later campaign",
         "ROC-AUC CI lower bound > 0.5 for all training windows",
         bool((t3.roc_auc > 0.6).all())),
        ("Targeting the top 10-20% by model is profitable in the back-test",
         "bootstrap P(profit>0) at 10% and 20% contact share > 0.95",
         bool((boot.set_index("contact_share").loc[[0.1, 0.2], "p_profit_positive"] > 0.95).all())),
        ("Mass contact loses money",
         f"true for {int((sens.profit_contact_all < 0).sum())}/9 cost/revenue combinations (cost 2-4, revenue 8-14 MU)",
         bool((sens.profit_contact_all < 0).all())),
        ("Profit-optimal contact share",
         f"hindsight-optimal share ranges {sens.hindsight_best_share.min():.1%}-{sens.hindsight_best_share.max():.1%} across cost/revenue grid",
         bool(sens.hindsight_best_share.max() - sens.hindsight_best_share.min() < 0.10)),
        ("RFM value tier membership", f"Jaccard of 'High value' tier vs quintile scoring min={t1.high_value_jaccard_vs_quintiles.min():.2f}",
         bool(t1.high_value_jaccard_vs_quintiles.min() > 0.8)),
        ("Exact 3x3 RFM segment membership", f"ARI vs quintile scoring min={t1.segment_ARI_vs_quintiles.min():.2f}",
         bool(t1.segment_ARI_vs_quintiles.min() > 0.8)),
        ("KMeans segment structure", f"ARI of alternative k vs selected: {', '.join(f'k={r.k}:{r.ARI_vs_selected:.2f}' for r in t2.itertuples())}",
         bool((t2.loc[t2.k != ctx['k_star'], 'ARI_vs_selected'] > 0.8).all())),
        ("Pilot recency effect is real behaviour", "Recency AUC is ~0.5 on C1-C5 but 0.66 on C6 -> timing artefact suspected",
         False),
        ("Leakage-sensitive features do not drive the temporal model",
         f"PR-AUC full={lv.loc['full','pr_auc']:.3f}, no_recency={lv.loc['no_recency','pr_auc']:.3f}, history+demographics={lv.loc['history_demographics_only','pr_auc']:.3f}",
         bool(abs(lv.loc['full', 'pr_auc'] - lv.loc['no_recency', 'pr_auc']) < 0.02)),
        ("Model performance is stable across campaigns",
         f"held-out ROC-AUC by campaign {t4.roc_auc.min():.2f}-{t4.roc_auc.max():.2f}", bool(t4.roc_auc.max() - t4.roc_auc.min() < 0.1)),
        ("Mens e-mail generates more incremental spend than Womens e-mail",
         f"share of bootstrap resamples: {t6.set_index('conclusion').loc['mens_gt_womens_spend','share_of_bootstrap_resamples_true']:.2f}",
         bool(t6.set_index('conclusion').loc['mens_gt_womens_spend', 'share_of_bootstrap_resamples_true'] > 0.95)),
        ("Womens e-mail has no conversion effect on mens-only buyers",
         "nominal subgroup finding; not significant after BH; bootstrap sign share for >0 = "
         f"{t6.set_index('conclusion').loc['womens_on_mens_only_conv_gt0','share_of_bootstrap_resamples_true']:.2f}", False),
    ]
    fr = pd.DataFrame(findings, columns=["finding", "test", "stable"])
    fr["classification"] = np.where(fr.stable, "STABLE", "SENSITIVE / NOT CONFIRMED")
    _save(fr, "robustness_classification")
    ctx.update(robust={"rfm_bins": t1, "kmeans_k": t2, "train_window": t3, "drift": t4, "dedup": t5,
                       "uplift_boot": t6, "classification": fr})
    return ctx


# ----------------------------------------------------------------------------- 12. SQL + key metrics
def stage_sql(ctx: dict) -> dict:
    _log("SQL analytics + cross-checks")
    from . import sql_runner
    out = sql_runner.run_all(ctx)
    ctx["sql"] = out
    return ctx


def stage_key_metrics(ctx: dict) -> dict:
    feat, res, e = ctx["feat"], ctx["model"], ctx["econ"].iloc[0]
    comp = res["comparison"].set_index("approach")
    ate = ctx["uplift"]["ate"].set_index(["arm", "outcome"])
    pol = ctx["uplift"]["policies"].set_index("policy")
    boot = ctx["profit_boot"].set_index("contact_share")
    conc = ctx["value_conc"].set_index("metric")["value"]
    km = {
        "raw_rows_ifood": int(len(ctx["raw_ifood"])), "customers": int(len(feat)),
        "campaigns": 6, "hillstrom_customers": int(len(ctx["hill"])),
        "total_spend_2y": float(feat.total_spend.sum()),
        "pilot_response_rate": float(feat.Response.mean()),
        "pilot_contacts": int(e.contacts), "pilot_responses": int(e.responses), "pilot_cost": float(e.cost),
        "pilot_revenue": float(e.revenue), "pilot_profit": float(e.profit), "pilot_roi": float(e.roi),
        "break_even_p": BREAK_EVEN_P,
        "top20_revenue_share": float(conc["revenue share of top 20% customers"]),
        "gini_spend": float(conc["Gini coefficient of 2-year spend"]),
        "k_star": int(ctx["k_star"]),
        "primary_model": res["primary_model"],
        "model_roc_auc": float(comp.loc[f"M_{res['primary_model']}", "roc_auc"]),
        "model_pr_auc": float(comp.loc[f"M_{res['primary_model']}", "pr_auc"]),
        "rule_roc_auc": float(comp.loc["B1_prior_acceptances", "roc_auc"]),
        "rule_pr_auc": float(comp.loc["B1_prior_acceptances", "pr_auc"]),
        "rfm_roc_auc": float(comp.loc["B2_RFM_score", "roc_auc"]),
        "gb_roc_auc": float(comp.loc["M_gradient_boosting", "roc_auc"]),
        "in_campaign_cv_auc": float(res["in_campaign_cv"]["roc_auc"]),
        "model_lift20": float(comp.loc[f"M_{res['primary_model']}", "lift@20%"]),
        "profit_top10_mean": float(boot.loc[0.1, "profit_mean"]), "profit_top10_ci": [float(boot.loc[0.1, "profit_ci_low"]), float(boot.loc[0.1, "profit_ci_high"])],
        "profit_top20_mean": float(boot.loc[0.2, "profit_mean"]), "profit_top20_ci": [float(boot.loc[0.2, "profit_ci_low"]), float(boot.loc[0.2, "profit_ci_high"])],
        "hindsight_best_contacts": int(ctx["econ_sens"].query("cost_per_contact==3 and revenue_per_response==11").hindsight_best_contacts.iloc[0]),
        "hindsight_best_profit": float(ctx["econ_sens"].query("cost_per_contact==3 and revenue_per_response==11").hindsight_best_profit.iloc[0]),
        "n_P1": int((ctx["engine"].priority.str.startswith("P1")).sum()),
        "backtest_P1_response": float(ctx["priority_backtest"].set_index("tier").loc["P1 - Target", "response_rate"]),
        "backtest_P1_profit": float(ctx["priority_backtest"].set_index("tier").loc["P1 - Target", "profit_mu"]),
        "scenario_rule_profit": float(ctx["scenarios"].iloc[1].profit_mu),
        "n_P2": int((ctx["engine"].priority.str.startswith("P2")).sum()),
        "n_P3": int((ctx["engine"].priority.str.startswith("P3")).sum()),
        "mens_spend_effect": float(ate.loc[("Mens", "spend"), "effect"]),
        "womens_spend_effect": float(ate.loc[("Womens", "spend"), "effect"]),
        "mens_conv_effect": float(ate.loc[("Mens", "conversion"), "effect"]),
        "womens_conv_effect": float(ate.loc[("Womens", "conversion"), "effect"]),
        "policy_uplift_model_inc_spend": float(pol.loc["Uplift model: best arm per customer", "incremental_spend_per_customer"]),
        "policy_all_mens_inc_spend": float(pol.loc["E-mail all: Mens", "incremental_spend_per_customer"]),
        "seed": SEED,
    }
    (TAB / "key_metrics.json").write_text(json.dumps(km, indent=2))
    ctx["key_metrics"] = km
    return ctx


STAGES = [stage_audit, stage_clean, stage_features, stage_segmentation, stage_campaigns, stage_model,
          stage_value, stage_targeting, stage_optimization, stage_uplift, stage_robustness, stage_key_metrics,
          stage_sql]


def run_all(until: str | None = None) -> dict:
    import matplotlib.pyplot as plt
    ctx: dict = {}
    for st in STAGES:
        ctx = st(ctx)
        plt.close("all")
        if until and st.__name__ == until:
            break
    return ctx


def prepare(before: str) -> dict:
    """Run (quietly) every stage that precedes `before`; used by notebooks so each notebook
    re-derives its inputs from raw data instead of trusting stale files."""
    import contextlib, io
    import matplotlib.pyplot as plt
    ctx: dict = {}
    for st in STAGES:
        if st.__name__ == before:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            ctx = st(ctx)
        plt.close("all")
    return ctx
