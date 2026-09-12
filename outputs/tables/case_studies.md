### Case 1 - High-value customer — customer ID 1763
*Selected by rule:* highest 2-yr spend among P1 customers.

1. **Profile:** income 87,679, age 26, tenure 11.1 months.
2. **Behaviour:** 28 purchases (web 25%, catalog 39%, store 36%); last purchase 62 days ago; 4 web visits last month.
3. **Segment:** High-value (cooling) (RFM 2-5-5); cluster 'High-value, full-price'.
4. **Campaign history (C1..C6):** 101011 -> accepted 4/6.
5. **Value:** 2-yr spend 2,524 MU (percentile 100); AOV 90.1 MU.
6. **Response behaviour:** pilot (C6) accepted.
7. **Prediction:** model p(next campaign) = 0.92 vs break-even 0.273; drivers up: campaign acceptances so far (+3.94); 2-yr spend (+1.03); income (+0.30); drivers down: accepted most recent campaign (-0.57); gold-product share (-0.23).
8. **Channel evidence:** Mixed: pilot response 24.3% (95% CI 21.2%-27.7%, n=679). (Purchase channel, not campaign delivery channel.)
9. **Recommendation:** P1 - Target -> Include in next offer - priority (responsive + high value).
10. **Why:** customers in this tier responded at 40.3% (95% CI 35.7%-45.1%) in the pilot back-test.
11. **Limitation:** observational; response is not proof the offer *caused* a purchase; the pilot economics (3/11 MU) are assumed to carry over.

### Case 2 - High-response customer outside the RFM high-value tier — customer ID 3725
*Selected by rule:* highest model p among P1 customers whose RFM value tier is not High.

1. **Profile:** income 84,865, age 53, tenure 13.7 months.
2. **Behaviour:** 15 purchases (web 13%, catalog 27%, store 60%); last purchase 1 days ago; 4 web visits last month.
3. **Segment:** Mid-value (recent) (RFM 5-3-5); cluster 'High-value, full-price'.
4. **Campaign history (C1..C6):** 110111 -> accepted 5/6.
5. **Value:** 2-yr spend 1,688 MU (percentile 94); AOV 112.5 MU.
6. **Response behaviour:** pilot (C6) accepted.
7. **Prediction:** model p(next campaign) = 0.98 vs break-even 0.273; drivers up: campaign acceptances so far (+4.96); 2-yr spend (+0.58); days since last purchase (+0.30); drivers down: accepted most recent campaign (-0.57); gold-product share (-0.24).
8. **Channel evidence:** Store: pilot response 6.7% (95% CI 5.3%-8.3%, n=1065). (Purchase channel, not campaign delivery channel.)
9. **Recommendation:** P1 - Target -> Include in next offer (responsive; lower basket value).
10. **Why:** customers in this tier responded at 40.3% (95% CI 35.7%-45.1%) in the pilot back-test.
11. **Limitation:** observational; response is not proof the offer *caused* a purchase; the pilot economics (3/11 MU) are assumed to carry over.

### Case 3 - At-risk (high value, lapsing) customer — customer ID 9010
*Selected by rule:* high value tier, lapsing recency tier, has accepted before; highest spend.

1. **Profile:** income 83,151, age 42, tenure 16.6 months.
2. **Behaviour:** 22 purchases (web 23%, catalog 32%, store 45%); last purchase 80 days ago; 2 web visits last month.
3. **Segment:** High-value (lapsing) (RFM 1-5-5); cluster 'High-value, full-price'.
4. **Campaign history (C1..C6):** 101011 -> accepted 4/6.
5. **Value:** 2-yr spend 2,346 MU (percentile 100); AOV 106.6 MU.
6. **Response behaviour:** pilot (C6) accepted.
7. **Prediction:** model p(next campaign) = 0.89 vs break-even 0.273; drivers up: campaign acceptances so far (+3.94); 2-yr spend (+0.94); income (+0.26); drivers down: accepted most recent campaign (-0.57); web visits last month (-0.19).
8. **Channel evidence:** Mixed: pilot response 24.3% (95% CI 21.2%-27.7%, n=679). (Purchase channel, not campaign delivery channel.)
9. **Recommendation:** P1 - Target -> Include in next offer - priority (responsive + high value).
10. **Why:** customers in this tier responded at 40.3% (95% CI 35.7%-45.1%) in the pilot back-test.
11. **Limitation:** observational; response is not proof the offer *caused* a purchase; the pilot economics (3/11 MU) are assumed to carry over.

### Case 4 - Model and rule disagree — customer ID 1501
*Selected by rule:* never accepted any campaign (rule says suppress) but highest model p.

1. **Profile:** income 160,803, age 32, tenure 22.8 months.
2. **Behaviour:** 29 purchases (web 0%, catalog 97%, store 3%); last purchase 21 days ago; 0 web visits last month.
3. **Segment:** Core high-value (recent) (RFM 4-5-5); cluster 'High-value, full-price'.
4. **Campaign history (C1..C6):** 000000 -> accepted 0/6.
5. **Value:** 2-yr spend 1,717 MU (percentile 94); AOV 59.2 MU.
6. **Response behaviour:** pilot (C6) did not accept.
7. **Prediction:** model p(next campaign) = 0.54 vs break-even 0.273; drivers up: income (+0.91); catalog purchase share (+0.81); 2-yr spend (+0.60); drivers down: meat share (-0.49); web visits last month (-0.31).
8. **Channel evidence:** Catalog: pilot response 31.6% (95% CI 19.1%-47.5%, n=38). (Purchase channel, not campaign delivery channel.)
9. **Recommendation:** P2 - Test cell only -> Randomised test cell with holdout (expected below break-even).
10. **Why:** customers in this tier responded at 14.9% (95% CI 10.2%-21.2%) in the pilot back-test.
11. **Limitation:** observational; response is not proof the offer *caused* a purchase; the pilot economics (3/11 MU) are assumed to carry over.

### Case 5 — Surprising campaign × segment: C6 (pilot) in 'Mid-value (recent)'
Campaign C6 (pilot) was accepted by 21.6% of 'Mid-value (recent)' (95% CI 16.7%-27.5%, n=222) vs 14.6% overall (index 1.48), whereas most campaigns over-index only in high-value segments. **Implication:** campaign content matters for who responds — a C6-style offer is the only observed route to lower-value customers. **Limitation:** offer contents of C1-C5 are not documented, so the *type* of offer cannot be identified; CI is wide.
