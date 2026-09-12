"""Integrity tests on pipeline outputs (run after `python run_pipeline.py`)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "dashboard"))
TAB, PROC = ROOT / "outputs" / "tables", ROOT / "data" / "processed"
KM = json.loads((TAB / "key_metrics.json").read_text())


def test_raw_preserved_and_schema():
    from src import ingestion
    raw = ingestion.load_ifood_raw(); h = ingestion.load_hillstrom_raw()
    assert raw.shape == (2240, 29) and h.shape == (64000, 12)


def test_cleaning_counts():
    feat = pd.read_csv(PROC / "customer_features.csv")
    assert len(feat) == KM["customers"] and feat.ID.is_unique
    from src.ingestion import IFOOD_SCHEMA
    feats = [c for c in IFOOD_SCHEMA if c not in ("ID", "Response", "Z_CostContact", "Z_Revenue")]
    assert not feat.duplicated(subset=feats).any()


def test_pilot_economics_formula():
    feat = pd.read_csv(PROC / "customer_features.csv")
    assert KM["pilot_profit"] == feat.Response.sum() * 11 - len(feat) * 3
    assert abs(KM["pilot_roi"] - KM["pilot_profit"] / (3 * len(feat))) < 1e-12


def test_rfm_scores_valid():
    r = pd.read_csv(TAB / "rfm_scores.csv")
    for c in "RFM":
        assert set(r[c].unique()) == {1, 2, 3, 4, 5}
        assert r[c].value_counts().max() - r[c].value_counts().min() <= 1  # equal-size quintiles
    # higher R = more recent
    assert r.groupby("R").recency_days.mean().is_monotonic_decreasing


def test_panel_has_no_future_history():
    from src.config import CAMPAIGNS
    feat = pd.read_csv(PROC / "customer_features.csv").set_index("ID")
    panel = pd.read_csv(PROC / "campaign_panel.csv")
    for k in range(2, 7):
        p = panel[panel.campaign_k == k].set_index("ID")
        expected = feat.loc[p.index, CAMPAIGNS[:k - 1]].sum(axis=1)
        assert (p.hist_n_prior == expected).all(), f"history for C{k} uses the wrong campaigns"
        assert (p.target == feat.loc[p.index, CAMPAIGNS[k - 1]]).all()
        assert (p.hist_last == feat.loc[p.index, CAMPAIGNS[k - 2]]).all()


def test_model_features_exclude_targets():
    import joblib
    m = joblib.load(ROOT / "outputs" / "models" / "response_model_temporal.joblib")
    bad = [f for f in m["features"] if f.startswith(("AcceptedCmp", "Response", "target", "campaign_k"))]
    assert not bad, bad


def test_sql_crosscheck_passed():
    cc = pd.read_csv(TAB / "sql_python_crosscheck.csv")
    assert cc.match.all() and len(cc) >= 20


def test_dashboard_numbers_match_pipeline():
    import data as D
    scores = D.table("model_test_scores_C6")
    ops = D.table("model_operating_points").set_index("operating_point")
    m = D.backtest_at_share(scores, "M_logistic_regression", 0.2)
    assert m["tp"] == ops.loc["top 20%", "tp"] and abs(m["precision"] - ops.loc["top 20%", "precision"]) < 1e-9
    scen = D.table("allocation_scenarios_backtest")
    assert scen.iloc[4].profit_mu == m["profit"]
    eng = D.engine()
    assert eng.priority.value_counts()["P1 - Target"] == KM["n_P1"]
    sel = D.budget_selection(eng, 300, ["P1 - Target"], ["High value", "Mid value", "Low value"])
    assert len(sel) == 100 and (sel.priority == "P1 - Target").all()


def test_allocation_totals():
    bc = pd.read_csv(TAB / "budget_curve_backtest.csv")
    assert (bc.contacts * 3 <= bc.budget_mu + 1e-9).all()
    full = bc[bc.contacts == KM["customers"]]
    assert len(full) > 0
    assert np.allclose(full.profit_mu, KM["pilot_profit"])


def test_hillstrom_effects_consistent():
    h = pd.read_csv(PROC / "hillstrom_clean.csv")
    ctrl = h[h.treatment == "Control"].spend.mean()
    assert abs((h[h.treatment == "Mens"].spend.mean() - ctrl) - KM["mens_spend_effect"]) < 1e-12


def test_targeting_engine_complete():
    eng = pd.read_csv(PROC / "targeting_engine.csv")
    assert len(eng) == KM["customers"] and eng.p_next_campaign.between(0, 1).all()
    assert eng[["priority", "action", "evidence_summary", "drivers_up_logodds"]].notna().all().all()
    # P1 = everyone with >=1 acceptance
    assert ((eng.n_accepted_all6 > 0) == eng.priority.str.startswith("P1")).all()
