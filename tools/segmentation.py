"""KMeans segmentation evaluated AGAINST the RFM baseline (it must earn its place)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .evaluation import cramers_v

# Behavioural variables only (no outcome of the pilot campaign -> segments can be used to target it).
# log1p for right-skewed spend/frequency; shares are bounded [0,1].
CLUSTER_FEATURES = ["Recency", "log_frequency", "log_monetary", "NumWebVisitsMonth",
                    "deals_share", "share_catalog", "share_wines", "prior_accept_rate"]


def cluster_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "Recency": feat["Recency"],
        "log_frequency": np.log1p(feat["total_purchases"]),
        "log_monetary": np.log1p(feat["total_spend"]),
        "NumWebVisitsMonth": feat["NumWebVisitsMonth"].clip(upper=feat["NumWebVisitsMonth"].quantile(.995)),
        "deals_share": feat["deals_share"].fillna(0),
        "share_catalog": feat["share_catalog"].fillna(0),
        "share_wines": feat["share_wines"].fillna(0),
        "prior_accept_rate": feat["n_accepted_prior5"] / 5,
    }, index=feat.index)
    return X[CLUSTER_FEATURES]


def evaluate_k(X: pd.DataFrame, ks=range(2, 9), n_boot: int = 20) -> pd.DataFrame:
    Z = StandardScaler().fit_transform(X)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
        ref = km.labels_
        aris = []
        for b in range(n_boot):
            idx = rng.integers(0, len(Z), len(Z))
            kb = KMeans(k, n_init=10, random_state=SEED + b + 1).fit(Z[idx])
            aris.append(adjusted_rand_score(ref, kb.predict(Z)))
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, ref, random_state=SEED,
                                                                                    sample_size=len(Z)),
                     "stability_ari_mean": np.mean(aris), "stability_ari_min": np.min(aris),
                     "min_cluster_share": np.bincount(ref).min() / len(ref)})
    return pd.DataFrame(rows)


def choose_k(eval_df: pd.DataFrame, min_ari: float = 0.8, min_share: float = 0.05) -> int:
    """Rule (declared before looking at segment content): among solutions with mean bootstrap
    ARI >= 0.8 and no cluster < 5% of customers, take the highest silhouette."""
    ok = eval_df[(eval_df.stability_ari_mean >= min_ari) & (eval_df.min_cluster_share >= min_share)]
    if ok.empty:
        ok = eval_df
    return int(ok.sort_values("silhouette", ascending=False).iloc[0]["k"])


def fit_kmeans(X: pd.DataFrame, k: int):
    scaler = StandardScaler().fit(X)
    km = KMeans(k, n_init=50, random_state=SEED).fit(scaler.transform(X))
    return km, scaler


def label_clusters(X: pd.DataFrame, labels: np.ndarray, feat: pd.DataFrame) -> dict:
    """Evidence-based labels generated from each cluster's standardised centroid:
    value level from log_monetary, then the most distinctive remaining trait."""
    Z = (X - X.mean()) / X.std()
    cent = Z.groupby(labels).mean()
    trait_names = {"Recency": ("recently active", "lapsing"), "log_frequency": ("frequent", "infrequent"),
                   "NumWebVisitsMonth": ("web-browsing", "low web visits"),
                   "deals_share": ("deal-driven", "full-price"),
                   "share_catalog": ("catalog-heavy", "low catalog"),
                   "share_wines": ("wine-focused", "low wine share"),
                   "prior_accept_rate": ("campaign-responsive", "campaign-unresponsive")}
    names = {}
    for c, row in cent.iterrows():
        m = row["log_monetary"]
        level = "High-value" if m > 0.5 else ("Low-value" if m < -0.5 else "Mid-value")
        rest = row.drop(["log_monetary", "log_frequency"])
        t = rest.abs().idxmax()
        val = rest[t]
        # Recency sign is inverted (low days = recent)
        if t == "Recency":
            trait = trait_names[t][0] if val < 0 else trait_names[t][1]
        else:
            trait = trait_names[t][0] if val > 0 else trait_names[t][1]
        names[c] = f"{level}, {trait}"
    # de-duplicate identical names
    seen = {}
    for c in sorted(names):
        n = names[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    return names


def compare_with_rfm(feat: pd.DataFrame, seg_a: str, seg_b: str, target: str = "Response") -> pd.DataFrame:
    """How much does each segmentation separate the outcome and value? (higher = more useful)."""
    rows = []
    for s in (seg_a, seg_b):
        tab = pd.crosstab(feat[s], feat[target])
        grp = feat.groupby(s)["total_spend"]
        eta2 = (grp.count() * (grp.mean() - feat["total_spend"].mean()) ** 2).sum() / \
               ((feat["total_spend"] - feat["total_spend"].mean()) ** 2).sum()
        rates = feat.groupby(s)[target].mean()
        rows.append({"segmentation": s, "n_segments": feat[s].nunique(),
                     "response_cramers_v": cramers_v(tab),
                     "response_rate_range": rates.max() - rates.min(),
                     "spend_eta_squared": eta2})
    return pd.DataFrame(rows)
