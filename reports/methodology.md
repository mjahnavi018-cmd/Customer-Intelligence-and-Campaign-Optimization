# Methodology

## Lineage
```
SOURCE (GitHub mirrors) → data/raw (read-only) → src/preprocessing (clean, logged) → data/processed/ifood_customers_clean.csv
  → src/feature_engineering (customer table, customer × campaign panel) → src/rfm, src/segmentation
  → src/campaign_analysis (campaigns, channels, campaign × segment, pilot economics)
  → src/response_model (temporal validation) → src/customer_value
  → src/targeting (engine, tiers, action matrix, case studies) → src/optimization (budget back-test)
  → src/uplift (Hillstrom randomised experiment) → src/robustness → src/sql_runner (SQLite + cross-checks)
  → outputs/tables, outputs/figures, outputs/models → dashboard/app.py
```
`run_pipeline.py` executes all stages; notebooks call the same stage functions (`src/pipeline.py`).
Random seed **42** is used everywhere (KMeans, bootstraps, model fitting, splits, random baselines).

## Definitions and formulas
| Quantity | Definition |
|---|---|
| Total spend (historical value) | sum of the six `Mnt*` columns (last 2 years, MU) — **not CLV** |
| Frequency | NumWebPurchases + NumCatalogPurchases + NumStorePurchases |
| AOV | total spend / frequency |
| Channel share | channel purchases / frequency; *dominant channel* = share ≥ 50 %, else "Mixed" |
| Category entropy | normalised Shannon entropy of the six category shares |
| Tenure | (2014-06-30 − Dt_Customer) in days |
| Pilot cost / revenue / profit | 3 × contacts / 11 × responses / revenue − cost |
| ROI | profit / cost · **Break-even response rate** = 3 / 11 = 0.273 |
| Acceptance index | segment acceptance rate / campaign's overall rate |
| Incremental effect (Hillstrom) | mean(outcome │ arm) − mean(outcome │ control); Wald CI (proportions) / Welch CI (spend) |
| Policy value | IPW estimate on the held-out half: Σ yᵢ·1[armᵢ = π(xᵢ)] / p(armᵢ) / n |

## RFM
Quintile scores via `rank(method='first')` (equal bins despite ties in F). Score edges are saved to
`outputs/tables/rfm_score_edges.csv`. Business grid: recency terciles × terciles of (F+M)/2 → 9 descriptive
segments; labels describe the scores and are then checked against profiles. Alternatives tested: 3, 4, 10 bins.

## Segmentation
Eight behavioural features (no pilot outcome): Recency, log1p frequency, log1p spend, web visits (capped at p99.5), deal
share, catalog share, wine share, prior acceptance rate; standardised. K ∈ 2…8 evaluated by inertia, silhouette,
bootstrap stability (ARI of 20 resample fits vs the full fit) and minimum cluster share. **Declared rule:** among
solutions with mean ARI ≥ 0.8 and every cluster ≥ 5 %, pick the highest silhouette. Clusters are labelled
automatically from standardised centroids (value level + most distinctive trait). Usefulness vs RFM: Cramér's V with
pilot response, response-rate range, η² of spend.

## Statistical testing
χ² tests of independence (Cramér's V as effect size), Mann–Whitney U with rank-biserial correlation and Welch CIs
for mean differences, Wilson intervals for proportions. Families of tests are corrected with **Benjamini–Hochberg**
(exploratory families: responder profiles, campaign × segment, Hillstrom subgroups, heterogeneity) or **Holm**
(the 9 confirmatory Hillstrom average effects). Heterogeneity of subgroup effects: Cochran's Q.

## Response model
* Panel: one row per (customer, campaign k), k = 2…6; history features from campaigns < k only.
* Train C2–C5 (8,076 rows), test C6 (2,019 customers). LOCO CV inside training for tuning.
* Candidates: logistic regression (C ∈ {0.01, 0.1, 1, 10}) and histogram gradient boosting (4 configs).
  Random forest was not added (same non-linear family as boosting; no evidence it would change the decision).
* **Declared choice rule:** logistic regression unless boosting's best LOCO PR-AUC exceeds it by > 0.01.
* Baselines: B0 overall rate (no ranking), B1 number of earlier acceptances (+0.5 × accepted last), B2 RFM score.
* Metrics: ROC-AUC, **PR-AUC (primary)**, lift and recall at 10/20/30 %, precision/recall/F1 and confusion matrix at
  operating points, back-tested profit. Paired bootstrap (1,000) for model-vs-baseline differences.
* Explainability: standardised LR coefficients, permutation importance (PR-AUC drop, 20 repeats), exact per-customer
  log-odds contributions in the engine. SHAP not used (LR contributions are already exact).
* Calibration: decile reliability table; prior-shift (constant log-odds offset) to a target base rate for forward scoring.

## Targeting framework
Five prioritisation methods back-tested on the pilot at 10/20/30 % budgets (profit, responses, historical spend of
responders). Chosen tiers (evidence-based, no arbitrary weights):
* **P1 Target** — accepted ≥ 1 campaign; ordered by model probability.
* **P2 Test cell only** — never accepted, top 10 % model probability among never-acceptors.
* **P3 Suppress** — the rest.
The same definition applied to the pilot (history C1–C5) gives tier response rates used as evidence. Actions combine
tier with the RFM value tier (action matrix).

## Allocation
Rank-and-cut over a budget grid (150 MU steps) for five rankings; named scenarios A–F; customer bootstrap of profit;
marginal returns per 5 % slice; cost ∈ {2, 3, 4} × revenue ∈ {8, 11, 14} sensitivity. A linear program was not used:
with one channel and a constant cost per contact, the optimum of a budget-constrained expected-profit problem is
exactly a rank-and-cut. Forward plans are labelled *model-expected*.

## Incremental response (Hillstrom)
Average effects per arm (visit, conversion, spend) + Mens-vs-Womens; 6 pre-specified subgroup dimensions × 2 arms ×
2 outcomes (BH). T-learner (one logistic model per arm) on a 50 % training split; Qini coefficients vs 100 random
rankings; policies (none, all-Mens, all-Womens, segment rule learned on train, uplift-model best arm) evaluated on the
held-out half with IPW and a 300-resample bootstrap.

## Robustness classification criteria
A finding is **STABLE** if it survives every variation tested for it (explicit criterion per row in
`outputs/tables/robustness_classification.csv`), otherwise **SENSITIVE / NOT CONFIRMED**.
