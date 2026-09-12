# Data availability matrix

Verified against the actual files (see `outputs/tables/data_quality_report.csv`). The project scope follows from this
table: anything marked **No** is not analysed, and the limitation is stated wherever it matters.

| Feature | Available? | Dataset | Data type | Coverage | Missingness | Usable for | Limitation |
|---|---|---|---|---|---|---|---|
| Customer ID | Yes | iFood | int | 2,240 rows → 2,019 unique customers after de-dup | 0 % | joins, engine, SQL | 182 raw exact duplicates + 19 conflicting-label groups under different IDs |
| Customer ID | No (row index only) | Hillstrom | – | – | – | – | row index used as `customer_idx` |
| Campaign ID | Yes (implicit, ordered) | iFood | 6 binary columns | C1…C5, pilot (C6) for every customer | 0 % | campaign analysis, temporal panel | offer contents not documented |
| Campaign ID / treatment | Yes | Hillstrom | categorical (Mens / Womens / No e-mail) | 64,000 | 0 % | causal effects, uplift | one campaign wave |
| Campaign date | **No** | – | – | – | – | order only (C1 → C6) | no calendar time, no fatigue analysis, no seasonal drift |
| Campaign exposure | Partial | iFood | – | pilot: all 2,240 contacted (case statement) | – | pilot economics | exposure to C1–C5 not recorded (acceptance ≠ response among exposed) |
| Campaign exposure | Yes | Hillstrom | randomised arm | 64,000 | 0 % | treatment effects | – |
| Campaign response | Yes | iFood | binary × 6 | 1.3 %–14.6 % positive | 0 % | response model, campaign × segment | label timing vs snapshot ambiguous (see leakage audit) |
| Response / conversion | Yes | Hillstrom | visit, conversion, spend | 2-week window | 0 % | incremental response | conversion rare (0.9 %) |
| Purchase amount | Aggregated | iFood | 6 category totals, last 2 years | all customers | 0 % | value, RFM (M) | not per transaction; window not aligned with enrolment |
| Purchase amount | Aggregated | Hillstrom | $ spent past 12 months | all | 0 % | covariate | – |
| Purchase date / transactions | **No** | – | – | – | – | – | no transaction-level RFM, no repeat-purchase timing, no CLV |
| Recency | Yes | iFood | days since last purchase (0–99) | all | 0 % | RFM (R) | uniform distribution; likely measured after the pilot |
| Purchase frequency | Yes | iFood | counts by channel + deals | all | 0 % | RFM (F), channel mix | deals overlap with channel counts |
| Channel | Purchase channel only | iFood | web / catalog / store counts | all | 0 % | channel volume vs value | **campaign delivery channel not recorded → no channel ROI** |
| Channel | Purchase channel + e-mail treatment | Hillstrom | phone / web / multichannel | all | 0 % | subgroup effects | only one treatment channel (e-mail) |
| Revenue | Per-response constant | iFood | 11 MU per pilot response (`Z_Revenue`) | pilot only | 0 % | pilot profit, allocation | campaign revenue, not margin; unknown for C1–C5 |
| Revenue | Yes | Hillstrom | spend $ in 2 weeks | all | 0 % | incremental revenue | revenue, not margin |
| Cost | Per-contact constant | iFood | 3 MU per contact (`Z_CostContact`) | pilot only | 0 % | ROI, break-even, budget | no cost for C1–C5, no channel costs, no CAC |
| Cost | **No** | Hillstrom | – | – | – | – | e-mail cost scenarios labelled HYPOTHETICAL |
| Engagement | Weak | iFood | web visits last month | all | 0 % | covariate | timing vs campaigns unknown; not a funnel stage |
| Engagement | Yes | Hillstrom | website visit after e-mail | all | 0 % | secondary outcome | – |
| Product / category | Yes | iFood | 6 category spends | all | 0 % | category mix, features | – |
| Product | Buyer type | Hillstrom | mens / womens purchase flags | all | 0 % | subgroup effects | – |
| Demographics | Yes | iFood | birth year, education, marital, income, kids | all | income 1.1 % missing (+1 invalid) | profiles, model | 3 impossible birth years; 7 odd marital labels |
| Customer tenure | Yes | iFood | enrolment date | 2012-07-30 → 2014-06-29 | 0 % | tenure feature, cohorts | reference date assumed = last enrolment + 1 day |
| Acquisition channel | **No** | – | – | – | – | – | "which channels acquire high-value customers" cannot be answered |
| Customer lifetime value | **No** | – | – | – | – | – | only *historical* value is reported |
| Treatment / control | iFood: **No**; Hillstrom: **Yes** | Hillstrom | – | – | – | uplift, policy value | iFood targeting is predictive, not causal |
| Complaints | Yes | iFood | binary (last 2 years) | 21 customers | 0 % | covariate | rare |

## Resulting scope
**In scope:** data audit and cleaning; SQL analytics; RFM and segmentation; campaign acceptance and campaign × segment;
purchase-channel volume vs value; pilot ROI/break-even; response prediction with campaign-ordered temporal validation;
historical value and concentration; targeting engine; budget allocation back-test; randomised incremental-response
analysis and policy evaluation (Hillstrom); robustness.

**Out of scope (data do not support):** channel ROI / ROAS / CAC, acquisition-channel value, campaign fatigue, calendar
drift, CLV / predicted future value, iFood uplift, funnel stages beyond contacted → accepted.
