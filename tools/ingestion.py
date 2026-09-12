"""Raw-data ingestion with schema checks.

Raw files are NOT redistributed in this repository (see DATA_SOURCES.md). Download them
manually and place them in data/raw/ with the exact filenames below, then run
    python -m src.ingestion --check
"""
from __future__ import annotations

import hashlib
import sys

import pandas as pd

from .config import (DATA_RAW, IFOOD_FILE, HILLSTROM_FILE, IFOOD_URL, HILLSTROM_URL,
                     IFOOD_EXPECTED_ROWS, HILLSTROM_EXPECTED_ROWS)

IFOOD_SCHEMA = ["ID", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome",
                "Teenhome", "Dt_Customer", "Recency", "MntWines", "MntFruits",
                "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
                "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
                "NumStorePurchases", "NumWebVisitsMonth", "AcceptedCmp3", "AcceptedCmp4",
                "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2", "Complain", "Z_CostContact",
                "Z_Revenue", "Response"]
HILLSTROM_SCHEMA = ["recency", "history_segment", "history", "mens", "womens", "zip_code",
                    "newbie", "channel", "segment", "visit", "conversion", "spend"]


class RawDataError(RuntimeError):
    pass


def _read(fname: str, url: str, schema: list[str], rows: int) -> pd.DataFrame:
    path = DATA_RAW / fname
    if not path.exists():
        raise RawDataError(f"Missing {path}.\nDownload from {url} (use 'Download raw file') "
                           f"and save it as data/raw/{fname}")
    # sep=None sniffs the delimiter; the project is validated on the original comma-separated file.
    df = pd.read_csv(path, sep=None, engine="python")
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise RawDataError(f"{fname}: missing expected columns {missing}")
    if len(df) != rows:
        raise RawDataError(f"{fname}: expected {rows} rows, found {len(df)}")
    return df[schema]


def load_ifood_raw() -> pd.DataFrame:
    return _read(IFOOD_FILE, IFOOD_URL, IFOOD_SCHEMA, IFOOD_EXPECTED_ROWS)


def load_hillstrom_raw() -> pd.DataFrame:
    return _read(HILLSTROM_FILE, HILLSTROM_URL, HILLSTROM_SCHEMA, HILLSTROM_EXPECTED_ROWS)


def sha256(fname: str) -> str:
    return hashlib.sha256((DATA_RAW / fname).read_bytes()).hexdigest()


if __name__ == "__main__":
    ok = True
    for loader, f in ((load_ifood_raw, IFOOD_FILE), (load_hillstrom_raw, HILLSTROM_FILE)):
        try:
            d = loader()
            print(f"OK  {f}: {d.shape[0]} rows x {d.shape[1]} cols  sha256={sha256(f)[:16]}...")
        except RawDataError as e:
            ok = False
            print(f"ERR {e}")
    sys.exit(0 if ok else 1)
