# Findings

Each finding: **Finding → Business implication → Recommended action → Evidence → Assumption → Limitation.**
All numbers are produced by `python run_pipeline.py` (tables in `outputs/tables/`). iFood numbers use the 2,019
de-duplicated customers; MU = monetary units of the case.

---

### F1. Past campaign acceptance is the strongest signal of future response — STABLE
* **Finding:** pilot response was 7.9 % for customers who had accepted none of C1–C5 (n = 1,602), 30.5 % after one
  acceptance, 50.7 % after two and 82 % after three or more.
* **Implication:** the 417 earlier acceptors are the natural audience; the other 1,602 customers mostly waste contacts.
* **Action:** make "accepted ≥ 1 campaign" the core of targeting (tier P1).
* **Evidence:** `repeat_response.csv` (Wilson CIs), top permutation importance in the model, rule ROC-AUC 0.72–0.73 on raw and de-duplicated data.
* **Assumption:** propensity to accept offers is a stable customer trait.
* **Limitation:** observational — repeat acceptors might have bought anyway (see F8).

### F2. Machine learning adds little over the one-line rule — model > rule NOT CONFIRMED
* **Finding:** trained on C2–C5 and tested on the later pilot, logistic regression reaches ROC-AUC 0.740 / PR-AUC 0.412;
  the "number of earlier acceptances" rule 0.728 / 0.396. Paired bootstrap PR-AUC difference +0.017 (95 % CI −0.013 to +0.043).
  Gradient boosting: ROC-AUC 0.766, PR-AUC 0.408 (no PR-AUC gain). RFM score: 0.722 / 0.283 (significantly worse than the model on PR-AUC).
  In the budget back-test the rule (with model tie-breaks) earns more profit at 10 % and 20 % budgets (571 / 570 MU vs 439 / 295 MU).
* **Implication:** a transparent rule is as good as the model for deciding *who*; the model's value is ordering and explanation.
* **Action:** rule-first tiers, model ordering within tiers; do not deploy boosting.
* **Evidence:** `model_comparison_test_C6.csv`, `model_paired_bootstrap_differences.csv`, `priority_method_comparison.csv`.
* **Assumption:** the pilot is representative of a "future" campaign.
* **Limitation:** a single test campaign; held-out AUC varies 0.65–0.88 across campaigns.

### F3. Naïve validation would have overstated performance (AUC 0.91 vs 0.74) — leakage artefact — STABLE
* **Finding:** Recency predicts only the pilot (AUC 0.66) and none of C1–C5 (0.48–0.54). Same-campaign CV gives AUC 0.915.
* **Implication:** the snapshot was most likely taken after the pilot; typical results reported for this dataset are inflated.
* **Action:** evaluate only with the campaign-ordered temporal design; treat pilot recency effects as unproven.
* **Evidence:** `leakage_single_feature_auc_by_campaign.csv`, `model_validation_designs.csv`, `model_leakage_variants.csv`.
* **Assumption:** a genuine recency effect would appear in every campaign.
* **Limitation:** snapshot timing is inferred, not documented.

### F4. Mass contact destroyed value; targeting a fifth of the base would have been profitable — STABLE (direction) / SENSITIVE (exact cut-off)
* **Finding:** the pilot contacted 2,019 customers: 294 responses, cost 6,057 MU, revenue 3,234 MU, **profit −2,823 MU (ROI −47 %)**.
  Contacting only the 417 earlier acceptors: 168 responses, **+597 MU**, 57 % of responders kept. Model top 20 %: +295 MU
  (bootstrap 95 % CI 97–526); P(profit > 0) > 0.99 at 10–20 % budgets. Mass contact loses money in 8 of 9 cost/revenue scenarios;
  the profit-optimal contact share ranges 5–30 % depending on the economics.
* **Implication:** budget should shrink to ≈ 1,250 MU (≈ 20 % of the pilot's).
* **Action:** contact P1 (543 customers for the next campaign, ≈ 1,629 MU); if the budget is smaller, fill it in P1 model order.
* **Evidence:** `pilot_economics.csv`, `allocation_scenarios_backtest.csv`, `allocation_profit_bootstrap_model.csv`, `economics_sensitivity.csv`.
* **Assumption:** 3 MU per contact and 11 MU per response carry over.
* **Limitation:** back-test on the campaign used to design the tiers (optimistic); revenue is campaign revenue, not margin.

### F5. Value is concentrated, and value ≠ responsiveness — STABLE
* **Finding:** top 20 % of customers = 52 % of 2-year spend (Gini 0.54). High-value tercile response 24.8 % vs 7.0 % low tercile,
  but "High-value (lapsing)" customers responded at only 15.1 % while spending as much as the core group (31.6 %).
  Ranking by historical value alone is less profitable than the rule; a probability/value blend raises the historical spend of
  captured responders by 14–30 % at a profit cost of 121–220 MU.
* **Implication:** there is a real trade-off between response volume and customer quality.
* **Action:** keep value as the second axis of the action matrix (priority offers to high-value P1; non-paid service for high-value
  non-responders), and choose the blend only if the business values basket size beyond the 11 MU per response.
* **Evidence:** `value_concentration.csv`, `value_tercile_response.csv`, `pilot_economics_by_rfm_segment.csv`, `fig15`.
* **Assumption:** 2-year spend proxies value.
* **Limitation:** historical value only — no CLV, no margins.

### F6. Campaign content decides who responds — campaign × segment interaction — STABLE (pattern) / wide CIs (small cells)
* **Finding:** C1, C2, C4, C5 over-index 2–3× in the three high-value segments and barely reach low-value customers
  (C5: 0 acceptors among low-value segments). C3 has no significant segment pattern (BH p = 0.06) and over-indexes in
  "Low-value (recent)" (index 1.31). C3 acceptors spend half as much as C5 acceptors (723 vs 1,601 MU).
* **Implication:** most offers appeal to affluent customers; a C3-type offer is the only observed route to lower-value customers.
* **Action:** match the offer to the tier; if the goal is to develop low-value customers, test a C3-style offer rather than premium offers.
* **Evidence:** `campaign_x_rfm_segment_index.csv`, `campaign_x_rfm_segment_tests.csv` (BH), `case_studies.md` (Case 5).
* **Assumption:** every customer was eligible for every campaign.
* **Limitation:** C1–C5 exposure and offer content unknown; no campaign dates → fatigue cannot be analysed.

### F7. The high-volume channel is not the high-value channel — association only
* **Finding:** Store = 46 % of purchases but store-dominant customers spend 402 MU and responded at 6.7 % (n = 1,065);
  catalog = 21 % of purchases, users spend 801 MU; catalog-dominant customers responded at 31.6 % (CI 19–48 %, n = 38).
* **Implication:** catalog usage marks valuable, responsive customers; store-only buyers are poor campaign targets.
* **Action:** use purchase-channel mix as a targeting attribute; **no budget shift between channels** is recommended.
* **Evidence:** `channel_summary.csv`, `channel_intensity_response.csv`.
* **Assumption:** purchase counts reflect channel usage.
* **Limitation:** campaign delivery channel, channel cost and channel revenue are not in the data — no channel ROI.

### F8. Randomised evidence: e-mails cause incremental revenue; Mens e-mail > Womens; personalisation does not pay — STABLE (ATE) / NOT CONFIRMED (subgroups)
* **Finding (Hillstrom, n = 64,000):** Mens e-mail +7.7 pp visits, +0.68 pp conversion, **+$0.77 spend/customer (CI 0.49–1.05)**;
  Womens +4.5 pp, +0.31 pp, **+$0.42 (0.17–0.68)**; Mens − Womens = +$0.35 (Holm p = 0.03; 98 % of bootstrap resamples).
  No heterogeneity survives BH (smallest p_BH = 0.16). The uplift-model policy (+$0.888/customer) equals "Mens to everyone" (+$0.888).
  Uplift ranking beats response ranking for Womens-e-mail visits (Qini 84 vs 51) but nothing beats random for rare conversions.
* **Implication:** *predicted response ≠ incremental response*, but at this scale choosing the best single campaign captures the value.
* **Action:** send the single best campaign; reserve uplift modelling for frequent outcomes or larger tests; run holdouts.
* **Evidence:** `hillstrom_average_effects.csv`, `hillstrom_heterogeneity_tests.csv`, `hillstrom_qini.csv`, `hillstrom_policy_values.csv`.
* **Assumption:** randomisation valid (balance SMD ≤ 0.014); 2-week window captures the effect.
* **Limitation:** different retailer (2008); spend is revenue; no e-mail cost (scenarios labelled HYPOTHETICAL).

### F9. KMeans adds little over RFM — segmentation NOT upgraded
* **Finding:** the pre-declared rule selects k = 2 (silhouette 0.30, ARI 0.98): high-value full-price vs low-value deal-driven.
  It separates pilot response less than RFM (Cramér's V 0.135 vs 0.267); alternative k give different partitions (ARI 0.33–0.51).
* **Action:** RFM stays the segmentation of record; the cluster is a descriptive attribute only.
* **Limitation:** k = 4 (also stable) isolates a campaign-responsive cluster, but only because prior acceptance is an input.

### F10. Data artefacts limit what can be concluded
Duplicates (183 + 38 conflicting rows), near-uniform recency, a spend window misaligned with enrolment, and no dates.
**Action:** treat all iFood results as describing this dataset; validate in production with a randomised holdout.

## Null / negative results (reported deliberately)
* Web visits are not associated with pilot response (BH p = 0.92).
* Gradient boosting does not improve PR-AUC; ML does not significantly beat the rule.
* Per-customer uplift assignment does not beat a single best campaign.
* KMeans does not improve on RFM.
* No channel ROI, CLV, campaign-fatigue or acquisition-channel analysis is possible with these data.
