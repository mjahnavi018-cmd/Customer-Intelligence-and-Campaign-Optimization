# Dataset evaluation

**Goal:** find public data that can support *customer value + segmentation + campaign response + channel +
cost/allocation* decisions, and — separately — *incremental (causal) response*. No single public dataset covers all
of these, so candidates were scored on the 23 criteria of the project brief and a two-dataset design was chosen
(analysed side by side, **never merged at row level** — they describe different companies and customers).

Facts marked † were verified by downloading and auditing the file in this project. Facts marked ‡ come from the
source's documentation / public notebooks; those datasets were not downloaded (they were not selected).

## Candidates

| # | Dataset | Customers / rows | Campaigns & response | Purchase / value | Channel | Cost | Time | Causal design | License |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **iFood CRM case** (a.k.a. Kaggle *Customer Personality Analysis*) † | 2,240 rows (2,019 after de-duplication) | 6 ordered campaigns, binary acceptance each (C1–C5 + pilot `Response`) | 2-year spend in 6 categories; purchase counts | purchase counts by web / catalog / store; web visits last month | **3 MU per contact, 11 MU revenue per response (pilot)** | enrolment dates 2012-07 → 2014-06; no transaction or campaign dates | none (observational) | GitHub repo has no license; Kaggle mirror lists CC0 |
| 2 | **Hillstrom MineThatData E-mail Challenge** † | 64,000 | 2 e-mail campaigns + no-e-mail control, **randomised**; visit / conversion / spend in 2 weeks | 12-month spend history (continuous + band) | past purchase channel (phone / web / multichannel); treatment channel = e-mail | none | single 2-week window (2008) | **randomised 1/3-1/3-1/3** | publicly released for a data-mining challenge; no explicit license |
| 3 | UCI Bank Marketing ‡ | 45,211 | telemarketing campaign; subscription yes/no; number of contacts, previous outcome | none (balance only) | contact type (cellular / telephone) | none | May 2008 – Nov 2010 (month/day only) | none | CC BY 4.0 |
| 4 | UCI Online Retail II ‡ | ~1.07 M transactions, ~5.9 k customers | **no campaigns** | full invoice history | none | none | 2009-12 → 2011-12 | none | CC BY 4.0 |
| 5 | Criteo Uplift ‡ | ~14 M | randomised ad exposure; visit / conversion | none | display ads | none | none | randomised | CC BY-NC-SA 4.0 |
| 6 | IBM Watson *Marketing Customer Value Analysis* ‡ | ~9.1 k (reported) | one renewal offer; Response; offer type 1–4 | premium, claims, a pre-computed "CLV" field | sales channel (agent / branch / call centre / web) | none | ~2 months | none (offer type not randomised) | IBM sample data (fictional) |

## Scoring (0 = absent, 1 = weak, 2 = usable, 3 = strong)

| Criterion | iFood | Hillstrom | Bank | Online Retail II | Criteo | IBM CVA |
|---|---|---|---|---|---|---|
| 1 Customer count | 1 | 3 | 3 | 2 | 3 | 2 |
| 2 Transaction count | 0 | 0 | 0 | 3 | 0 | 0 |
| 3 Campaign count | 3 | 2 | 1 | 0 | 1 | 1 |
| 4 Response labels | 3 | 3 | 3 | 0 | 3 | 2 |
| 5 Purchase history | 2 | 1 | 0 | 3 | 0 | 1 |
| 6 Channel information | 1 (purchase channel) | 1 | 1 | 0 | 0 | 2 |
| 7 Revenue | 1 (per-response constant) | 2 (spend) | 0 | 3 | 0 | 1 |
| 8 Cost | 2 (per contact, pilot only) | 0 | 0 | 0 | 0 | 0 |
| 9 Demographics | 3 | 1 | 3 | 0 | 0 | 3 |
| 10 Engagement | 1 (web visits) | 2 (visits) | 1 | 0 | 2 | 0 |
| 11 Time coverage | 1 | 0 | 2 | 3 | 0 | 1 |
| 12 Longitudinal structure | 2 (campaign order) | 0 | 1 | 3 | 0 | 0 |
| 13 Missingness (3 = little) | 3 | 3 | 2 | 2 | 3 | 3 |
| 14 Data quality | 2 (duplicates, artefacts) | 3 | 3 | 2 | 3 | 2 (synthetic) |
| 15 Class balance | 2 (1–15 %) | 1 (0.9 % conv.) | 2 | – | 1 | 2 |
| 16 Segmentation suitability | 3 | 1 | 1 | 3 | 0 | 2 |
| 17 Response-model suitability | 3 | 2 | 3 | 0 | 2 | 2 |
| 18 Customer-value suitability | 2 | 1 | 0 | 3 | 0 | 1 |
| 19 Optimisation suitability | 2 | 2 | 1 | 0 | 2 | 1 |
| 20 License | 2 | 1 | 3 | 3 | 1 | 1 |
| 21 Accessibility | 2 | 2 | 3 | 3 | 1 | 2 |
| 22 Reproducibility | 3 | 3 | 3 | 3 | 2 | 2 |
| 23 Documentation | 2 (data dictionary + case) | 2 (challenge post) | 3 | 3 | 2 | 1 |

## Decision

* **Primary — iFood.** The only candidate that combines customer value, behaviour, *several* campaigns per customer
  (which enables a campaign-ordered temporal validation), purchase channels **and real per-contact cost and
  per-response revenue** for one campaign, so profit and budget allocation can be computed without inventing costs.
* **Secondary — Hillstrom.** The only small, interpretable dataset with **randomised** treatment and a *choice between
  campaign types*. It answers the questions iFood cannot: *incremental* response and *which campaign for whom*.
* **Rejected:** Bank Marketing (no value/revenue, one product), Online Retail II (no campaigns), Criteo (anonymised
  features, no value, non-commercial license), IBM CVA (fictional sample data with a pre-computed CLV of unknown
  method and non-randomised offers — would invite exactly the causal and CLV over-claims the brief forbids).
* **Combining:** iFood and Hillstrom share no customers, time period or company → **not merged**. Hillstrom findings
  are presented as *evidence about campaign-type choice and incrementality*, not as facts about iFood customers.

## Access note
The analysis environment could not reach dataset hosts directly; both files were downloaded through a browser from
the GitHub mirrors listed in `DATA_SOURCES.md` and verified (row counts, schema, checksums of the files used).
