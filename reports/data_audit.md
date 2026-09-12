# Data audit

Produced by `notebooks/01_data_audit.ipynb` / `src/data_validation.py`. Machine-readable outputs:
`outputs/tables/data_quality_report.csv` (column profile) and `outputs/tables/data_quality_rule_checks.csv`.

## iFood CRM dataset — 2,240 rows × 29 columns

| Check | Result | Treatment |
|---|---|---|
| Schema / types | 25 int, 1 float (Income), 3 text (Education, Marital_Status, Dt_Customer) | Dt_Customer parsed to date |
| Duplicate IDs | 0 | – |
| Rows identical on all fields except ID | **182** (183 after marital-label normalisation) | duplicate registrations → keep lowest ID |
| Identical on all features but conflicting `Response` | **19 groups / 38 rows** | excluded: true label unknowable |
| Income missing | 24 (1.1 %) | NaN in clean table; median imputation inside model pipelines (training data only) |
| Income = 666,666 | 1 | placeholder → NaN + flag |
| Year_Birth 1893 / 1899 / 1900 | 3 | age → NaN + flag |
| Marital_Status 'Alone' (3), 'Absurd' (2), 'YOLO' (2) | 7 | Single / Unknown |
| Education '2n Cycle' vs 'Master' | ambiguous coding (203 rows) | kept, documented |
| Dt_Customer range | 2012-07-30 → 2014-06-29 | tenure reference = 2014-06-30 (assumption) |
| Z_CostContact / Z_Revenue | constant 3 / 11 | used as pilot economics, removed as features |
| Zero purchases (web+catalog+store) | 6 customers | kept, flagged via `dominant_channel = No purchases` |
| Deals > total channel purchases | 3 | kept (minor inconsistency) |
| Negative amounts | 0 | – |
| Spend distribution | median 396, max 2,525 MU; strongly right-skewed | log1p for clustering; ranks for RFM |
| Recency distribution | ≈ uniform 0–99 days (decile-count CV ≈ 0.04) | **provenance caveat**: unusual for real CRM data |
| Class balance | C1 6.4 %, C2 1.3 %, C3 7.3 %, C4 7.5 %, C5 7.3 %, pilot 14.9 % (raw) | PR-AUC / lift / profit metrics |
| Complain | 21 customers | kept |

**Suspicious artefacts**
1. *Recency vs pilot:* Recency has ROC-AUC ≈ 0.5 for C1–C5 but 0.66 for the pilot → the snapshot was most likely taken
   after the pilot (a pilot purchase resets Recency). See `reports/leakage_audit.md`.
2. *Spend window vs tenure:* customers enrolled 1.5 months before the snapshot already show 474 MU of "last 2 years"
   spend on average (≈ 323 MU per tenure month vs 38 for the oldest cohort) → spend is not aligned with enrolment.
3. *Duplicates with conflicting labels* indicate record-level noise in the outcome.

**Unique counts:** 2,019 customers after cleaning; 6 campaigns; **no transactions** (spend is aggregated) and no
campaign or transaction dates.

## Hillstrom — 64,000 rows × 12 columns
| Check | Result |
|---|---|
| Missing values | none |
| Arm sizes | Mens 21,307 · Womens 21,387 · No e-mail 21,306 |
| Randomisation balance | max absolute standardised mean difference 0.014 (all covariates) |
| spend > 0 ⇔ conversion = 1 | consistent (0 violations) |
| Category typo | 'Surburban' (28,776 rows) → 'Suburban' |
| Fully identical rows | 6,562 — expected: 12 low-cardinality fields and no ID; kept |
| Conversion rate | 0.9 % (rare) — visit (14.7 %) used as a secondary, better-powered outcome |

## Consequence for scope
See `DATA_AVAILABILITY.md`. The audit removes channel ROI, fatigue, CLV and iFood-uplift from scope and makes the
leakage audit mandatory before any response model is trusted.
