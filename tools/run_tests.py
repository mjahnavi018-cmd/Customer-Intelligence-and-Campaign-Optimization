"""Run all tests without pytest:  python tests/run_tests.py   (pytest tests/ also works)."""
import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
failed = 0
for f in sorted(HERE.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        t = time.time()
        try:
            getattr(mod, name)()
            print(f"PASS {f.stem}.{name} ({time.time()-t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {f.stem}.{name}\n{traceback.format_exc()}")
print("ALL TESTS PASSED" if not failed else f"{failed} TEST(S) FAILED")
sys.exit(1 if failed else 0)
