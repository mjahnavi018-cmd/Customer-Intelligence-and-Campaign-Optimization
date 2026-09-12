"""Run the complete analysis: python run_pipeline.py"""
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from src.pipeline import run_all

if __name__ == "__main__":
    t = time.time()
    ctx = run_all()
    km = ctx["key_metrics"]
    print("\n=== KEY METRICS ===")
    for k, v in km.items():
        print(f"{k:35s} {v}")
    print(f"\nDone in {time.time() - t:.0f}s. Outputs in outputs/, processed data in data/processed/.")
