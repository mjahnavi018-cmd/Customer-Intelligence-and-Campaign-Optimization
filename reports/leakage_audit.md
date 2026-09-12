# Leakage audit — response prediction

**Prediction task:** will a customer accept the *next* campaign, using only information available before it?

## 1. Timeline available in the data
Campaigns are ordered C1 → C5 → pilot (C6) but undated. All behavioural fields (spend, purchases, recency, web visits)
exist only as **one snapshot** whose date is not documented. Enrolment dates run to 2014-06-29.

## 2. Feature-by-feature classification

| Feature group | Available before campaign k? | Risk | Treatment |
|---|---|---|---|
| `AcceptedCmp_j`, j < k | yes | none | used as `hist_n_prior`, `hist_last`, `hist_ever` built **only from campaigns 1..k−1** (unit-tested in `tests/test_pipeline.py::test_panel_has_no_future_history`) |
| `AcceptedCmp_j`, j ≥ k / `Response` | no (future / target) | direct leakage | never used as features; test asserts no `AcceptedCmp*`/`Response` in the feature list |
| Demographics, Income, kids, education, marital | yes (slow-moving) | low | used |
| Tenure (`Dt_Customer`) | yes | low | used (reference date = last enrolment + 1 day) |
| `Recency` | **ambiguous** — snapshot likely after the pilot | **high for C6** | evidence below; model evaluated with and without it |
| 2-year spend, purchase counts, shares, AOV, deals, web visits | ambiguous — window may include the pilot purchase and purchases after C1–C5 | medium | evaluated with and without the whole group |
| `Complain` | "last 2 years" — ambiguous | low | used; negligible importance |
| `Z_CostContact`, `Z_Revenue` | constants | none | removed |

## 3. Evidence of timing leakage (Recency)
Single-feature ROC-AUC of −Recency (`outputs/tables/leakage_single_feature_auc_by_campaign.csv`):

| C1 | C2 | C3 | C4 | C5 | **C6 (pilot)** |
|---|---|---|---|---|---|
| 0.53 | 0.53 | 0.54 | 0.48 | 0.50 | **0.66** |

A genuine "recently active customers respond more" effect would appear in every campaign. It appears only in the
campaign closest to the snapshot, which is what a post-campaign snapshot (the accepted offer is itself a purchase)
would produce.

## 4. Design that neutralises it
* **Campaign-ordered temporal validation:** train on the customer × campaign panel for C2–C5, test on C6.
  Hyper-parameters chosen by leave-one-campaign-out CV inside C2–C5; the test campaign is never used for choices.
* **Imputation/scaling** fitted inside the sklearn pipeline on training rows only. No resampling is used (class
  imbalance is handled by ranking metrics; the business decision is a top-k contact list).
* **Sensitivity (same LR, same C):**

| Feature set | ROC-AUC (C6) | PR-AUC (C6) |
|---|---|---|
| full | 0.740 | 0.412 |
| without Recency | 0.742 | 0.414 |
| history + demographics only (no spend/behaviour, no Recency) | 0.746 | 0.462 |

The temporal model does **not** depend on the suspicious features.

## 5. What a naïve design would have reported
5-fold stratified CV *within* the pilot (the common approach for this dataset): ROC-AUC **0.915**, PR-AUC 0.66 —
versus the honest temporal 0.740 / 0.41. The gap is the combined effect of the Recency artefact and of evaluating on
the same campaign the model is fitted to. This figure is reported only as a warning.

## 6. Forward scoring
The engine scores the *next* (7th) campaign with a model trained on C2–C6. For that campaign every snapshot field
(including Recency) genuinely pre-dates it, so using them is legitimate. Probabilities are prior-shifted to the pilot
base rate — an explicit assumption, not a measured fact.

## 7. Residual risks
* Snapshot spend may include post-campaign purchases for C1–C5 training rows (features slightly "from the future"
  relative to those labels). The history + demographics variant shows conclusions do not change.
* Only six campaigns; campaign content differs (held-out AUC 0.65–0.88) — generalisation to a new offer is uncertain.
