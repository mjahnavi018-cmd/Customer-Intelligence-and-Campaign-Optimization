"""Headless smoke test of every dashboard page and several filter combinations.
Requires the pipeline outputs (python run_pipeline.py). Real Streamlit is replaced by a stub."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from streamlit_stub import make_stub, StopApp  # noqa: E402

APP = ROOT / "dashboard" / "app.py"
PAGES = ["Executive overview", "Campaign performance", "Channel analytics", "Customer segmentation", "Customer value",
         "Campaign response model", "Targeting engine", "Marketing allocation", "Customer investigation",
         "Model vs rule performance"]

SCENARIOS = [
    {},
    {"Segmentation": "KMeans cluster"},
    {"View": "KMeans (evaluated alternative)", "Compare segments on": "pilot_response_rate"},
    {"Channel": "Catalog"},
    {"Filter RFM segments": ["Core high-value (recent)"]},
    {"Probability threshold": 0.6},
    {"Budget (MU)": 300, "Tiers": ["P1 - Target", "P2 - Test cell only"], "Value tiers": ["High value"], "Min. probability": 0.5},
    {"Budget (MU)": 0},
    {"Budget (MU)": 4500, "Cost per contact (MU)": 2.0, "Revenue per response (MU)": 14.0, "Ranking": "M_logistic_regression",
     "HYPOTHETICAL cost per e-mail ($)": 0.5},
    {"Contact the top … of customers": 0.05},
]


def run_page(page, extra):
    import pandas as pd
    st = make_stub({"Page": page, **extra})
    eng = pd.read_csv(ROOT / "data" / "processed" / "targeting_engine.csv")
    st_ids = {"Customer ID": int(eng.ID.iloc[-1]), "Customer ID (ordered by priority)": int(eng.ID.iloc[5])}
    for k, v in st_ids.items():
        extra.setdefault(k, v)
    st = make_stub({"Page": page, **extra})
    sys.modules["streamlit"] = st
    try:
        runpy.run_path(str(APP), run_name="__main__")
    except StopApp:
        pass
    assert not st.errors, st.errors
    kinds = {k for k, _ in st.log}
    return len(st.log), kinds


def test_all_pages():
    results = []
    for page in PAGES:
        for sc in SCENARIOS:
            n, kinds = run_page(page, dict(sc))
            assert n > 3, f"{page} rendered almost nothing"
            results.append((page, n))
    return results


if __name__ == "__main__":
    res = test_all_pages()
    for page in PAGES:
        print(f"OK  {page:28s} elements rendered (default scenario): {[n for p, n in res if p == page][0]}")
    print(f"Dashboard smoke test passed: {len(PAGES)} pages x {len(SCENARIOS)} widget scenarios.")
