"""Project-wide paths, constants and assumptions (single source of truth)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
MODELS = OUT / "models"
SQL_DIR = ROOT / "sql"
DB_PATH = DATA_PROCESSED / "marketing.db"

for _p in (DATA_RAW, DATA_PROCESSED, FIG, TAB, MODELS, TAB / "sql"):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42  # used by every stochastic step (KMeans, bootstraps, models, splits)

# ---------------------------------------------------------------- raw files
IFOOD_FILE = "ml_project1_data.csv"
HILLSTROM_FILE = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"

IFOOD_URL = ("https://github.com/nailson/ifood-data-business-analyst-test/blob/master/"
             "ml_project1_data.csv")
HILLSTROM_URL = ("https://github.com/bscan/uplift/blob/master/"
                 "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

IFOOD_EXPECTED_ROWS = 2240
HILLSTROM_EXPECTED_ROWS = 64000

# ---------------------------------------------------------------- campaigns
# iFood data dictionary: AcceptedCmpN = accepted the offer in the N-th campaign;
# Response = accepted the offer in the LAST (6th, pilot) campaign.
CAMPAIGNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4",
             "AcceptedCmp5", "Response"]
CAMPAIGN_LABELS = {c: f"C{i+1}" for i, c in enumerate(CAMPAIGNS)}
CAMPAIGN_LABELS["Response"] = "C6 (pilot)"

# Economics of the pilot campaign, taken from the data itself (constant columns
# Z_CostContact = 3, Z_Revenue = 11) and consistent with the case statement:
# 2,240 contacts x 3 MU = 6,720 MU cost; 334 responses x 11 MU = 3,674 MU revenue.
# MU = "monetary units" (currency not disclosed). These apply ONLY to the pilot.
COST_PER_CONTACT = 3.0
REVENUE_PER_RESPONSE = 11.0
BREAK_EVEN_P = COST_PER_CONTACT / REVENUE_PER_RESPONSE  # 0.2727

MNT_COLS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
            "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLS = {"Web": "NumWebPurchases", "Catalog": "NumCatalogPurchases",
                "Store": "NumStorePurchases"}
