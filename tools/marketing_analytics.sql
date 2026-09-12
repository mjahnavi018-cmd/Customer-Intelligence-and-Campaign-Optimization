-- =====================================================================================
-- Marketing Campaign & Customer Analytics — SQL layer (SQLite >= 3.25 dialect)
-- Tables are built by src/sql_runner.py from data/processed/ (see schema at the bottom of
-- this header). Each query is delimited by "-- name: <id>" and answers ONE business question.
--
--   customers(ID, total_spend, total_purchases, web_purchases, catalog_purchases,
--             store_purchases, recency_days, income, dt_customer, rfm_segment,
--             value_tier, activity_tier, cluster_label, dominant_channel, response_c6)
--   campaign_responses(ID, campaign_no, campaign_label, accepted)   -- 1 row per customer x campaign
--   hillstrom(customer_idx, treatment, buyer_type, history, recency, channel, zip_code,
--             newbie, visit, conversion, spend)
--   targeting(ID, p_next_campaign, expected_profit_mu, priority, action)
-- Economics of the pilot (C6) come from the data: 3 MU per contact, 11 MU per response.
-- =====================================================================================

-- name: q01_revenue_concentration_deciles
-- Q: How concentrated is revenue among customers? (NTILE + window running share)
WITH ranked AS (
    SELECT ID, total_spend,
           NTILE(10) OVER (ORDER BY total_spend DESC, ID) AS value_decile
    FROM customers
), dec AS (
    SELECT value_decile, COUNT(*) AS customers, SUM(total_spend) AS revenue,
           MIN(total_spend) AS min_spend, MAX(total_spend) AS max_spend
    FROM ranked GROUP BY value_decile
)
SELECT value_decile, customers, revenue, min_spend, max_spend,
       ROUND(1.0 * revenue / SUM(revenue) OVER (), 4) AS revenue_share,
       ROUND(1.0 * SUM(revenue) OVER (ORDER BY value_decile ROWS UNBOUNDED PRECEDING)
             / SUM(revenue) OVER (), 4) AS cumulative_revenue_share
FROM dec ORDER BY value_decile;

-- name: q02_top_customers_by_revenue
-- Q: Which customers generate the most revenue, and did they respond to campaigns?
WITH acc AS (
    SELECT ID, SUM(accepted) AS campaigns_accepted
    FROM campaign_responses GROUP BY ID
)
SELECT c.ID, c.total_spend,
       RANK() OVER (ORDER BY c.total_spend DESC) AS revenue_rank,
       ROUND(100.0 * c.total_spend / (SELECT SUM(total_spend) FROM customers), 3) AS pct_of_revenue,
       c.rfm_segment, c.dominant_channel, a.campaigns_accepted
FROM customers c JOIN acc a USING (ID)
ORDER BY c.total_spend DESC
LIMIT 15;

-- name: q03_campaign_acceptance
-- Q: Which campaigns generate the most responses?
SELECT campaign_no, campaign_label,
       COUNT(*) AS customers, SUM(accepted) AS acceptors,
       ROUND(1.0 * SUM(accepted) / COUNT(*), 4) AS acceptance_rate,
       RANK() OVER (ORDER BY 1.0 * SUM(accepted) / COUNT(*) DESC) AS rank_by_rate
FROM campaign_responses
GROUP BY campaign_no, campaign_label
ORDER BY campaign_no;

-- name: q04_repeat_responders
-- Q: Which customers responded to repeated campaigns, and how valuable are they?
WITH per_customer AS (
    SELECT ID, SUM(accepted) AS n_accepted
    FROM campaign_responses GROUP BY ID
)
SELECT CASE WHEN n_accepted = 0 THEN '0 campaigns'
            WHEN n_accepted = 1 THEN '1 campaign'
            WHEN n_accepted = 2 THEN '2 campaigns'
            ELSE '3+ campaigns' END AS acceptance_group,
       COUNT(*) AS customers,
       ROUND(AVG(c.total_spend), 1) AS avg_spend,
       ROUND(1.0 * SUM(c.total_spend) / (SELECT SUM(total_spend) FROM customers), 4) AS revenue_share
FROM per_customer p JOIN customers c USING (ID)
GROUP BY acceptance_group ORDER BY acceptance_group;

-- name: q05_prior_acceptance_vs_next_campaign
-- Q: Does accepting campaign k predict accepting campaign k+1? (LEAD over campaign order)
WITH seq AS (
    SELECT ID, campaign_no, accepted,
           LEAD(accepted) OVER (PARTITION BY ID ORDER BY campaign_no) AS accepted_next
    FROM campaign_responses
)
SELECT campaign_no AS campaign_k,
       SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) AS acceptors_k,
       ROUND(AVG(CASE WHEN accepted = 1 THEN accepted_next END), 4) AS next_rate_if_accepted_k,
       ROUND(AVG(CASE WHEN accepted = 0 THEN accepted_next END), 4) AS next_rate_if_not_accepted_k
FROM seq WHERE accepted_next IS NOT NULL
GROUP BY campaign_no ORDER BY campaign_no;

-- name: q06_campaign_by_segment_rank
-- Q: Which campaigns perform best for which RFM segment? (rank segments within each campaign)
WITH r AS (
    SELECT cr.campaign_label, c.rfm_segment, COUNT(*) AS n,
           1.0 * SUM(cr.accepted) / COUNT(*) AS rate
    FROM campaign_responses cr JOIN customers c USING (ID)
    GROUP BY cr.campaign_label, c.rfm_segment
)
SELECT campaign_label, rfm_segment, n, ROUND(rate, 4) AS rate,
       ROUND(rate / AVG(rate) OVER (PARTITION BY campaign_label), 3) AS index_vs_segment_avg,
       DENSE_RANK() OVER (PARTITION BY campaign_label ORDER BY rate DESC) AS segment_rank
FROM r ORDER BY campaign_label, segment_rank;

-- name: q07_pilot_economics_by_segment
-- Q: Which segments would have made the pilot profitable? (actual cost 3 / revenue 11)
SELECT rfm_segment, COUNT(*) AS contacts, SUM(response_c6) AS responses,
       ROUND(1.0 * SUM(response_c6) / COUNT(*), 4) AS response_rate,
       3.0 * COUNT(*) AS cost_mu, 11.0 * SUM(response_c6) AS revenue_mu,
       11.0 * SUM(response_c6) - 3.0 * COUNT(*) AS profit_mu,
       CASE WHEN 11.0 * SUM(response_c6) > 3.0 * COUNT(*) THEN 'profitable' ELSE 'loss-making' END AS verdict
FROM customers GROUP BY rfm_segment ORDER BY profit_mu DESC;

-- name: q08_channel_volume_vs_value
-- Q: Which purchase channel has the volume and which has the value?
WITH ch AS (
    SELECT 'Web' AS channel, ID, web_purchases AS purchases, total_spend FROM customers
    UNION ALL SELECT 'Catalog', ID, catalog_purchases, total_spend FROM customers
    UNION ALL SELECT 'Store', ID, store_purchases, total_spend FROM customers
)
SELECT channel, SUM(purchases) AS purchases,
       ROUND(1.0 * SUM(purchases) / (SELECT SUM(purchases) FROM ch), 4) AS purchase_share,
       SUM(CASE WHEN purchases > 0 THEN 1 ELSE 0 END) AS customers_using,
       ROUND(AVG(CASE WHEN purchases > 0 THEN total_spend END), 1) AS avg_spend_of_users
FROM ch GROUP BY channel ORDER BY purchases DESC;

-- name: q09_dominant_channel_response
-- Q: Do customers who mainly buy in a given channel respond differently to the pilot?
SELECT dominant_channel, COUNT(*) AS customers,
       ROUND(AVG(total_spend), 1) AS avg_spend,
       ROUND(1.0 * SUM(response_c6) / COUNT(*), 4) AS pilot_response_rate
FROM customers GROUP BY dominant_channel ORDER BY pilot_response_rate DESC;

-- name: q10_enrolment_trend_rolling
-- Q: How do new-customer volume and spend change over time? (date functions + rolling 3-month mean)
WITH m AS (
    SELECT strftime('%Y-%m', dt_customer) AS enrol_month, COUNT(*) AS new_customers,
           AVG(total_spend) AS avg_spend, AVG(response_c6) AS pilot_response
    FROM customers GROUP BY enrol_month
)
SELECT enrol_month, new_customers, ROUND(avg_spend, 1) AS avg_spend,
       ROUND(AVG(new_customers) OVER (ORDER BY enrol_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS new_customers_3m_avg,
       ROUND(AVG(avg_spend) OVER (ORDER BY enrol_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS avg_spend_3m_avg,
       ROUND(pilot_response, 4) AS pilot_response
FROM m ORDER BY enrol_month;

-- name: q11_tenure_value
-- Q: Do longer-tenured customers spend more per month? (julianday date arithmetic)
WITH t AS (
    SELECT ID, total_spend,
           (julianday((SELECT MAX(dt_customer) FROM customers)) - julianday(dt_customer) + 1) / 30.44 AS tenure_months
    FROM customers
)
SELECT CASE WHEN tenure_months < 6 THEN '0-6m' WHEN tenure_months < 12 THEN '6-12m'
            WHEN tenure_months < 18 THEN '12-18m' ELSE '18-24m' END AS tenure_band,
       COUNT(*) AS customers, ROUND(AVG(total_spend), 1) AS avg_total_spend,
       ROUND(AVG(total_spend / MAX(tenure_months, 1)), 1) AS avg_spend_per_month
FROM t GROUP BY tenure_band ORDER BY MIN(tenure_months);

-- name: q12_hillstrom_arms
-- Q: What did each e-mail campaign do vs no e-mail? (randomised experiment)
WITH a AS (
    SELECT treatment, COUNT(*) AS n, AVG(visit) AS visit_rate, AVG(conversion) AS conv_rate,
           AVG(spend) AS spend_per_customer
    FROM hillstrom GROUP BY treatment
)
SELECT a.treatment, a.n, ROUND(a.visit_rate, 4) AS visit_rate, ROUND(a.conv_rate, 5) AS conversion_rate,
       ROUND(a.spend_per_customer, 4) AS spend_per_customer,
       ROUND(a.spend_per_customer - c.spend_per_customer, 4) AS incremental_spend_vs_control,
       ROUND(a.conv_rate - c.conv_rate, 5) AS incremental_conversion_vs_control
FROM a CROSS JOIN (SELECT * FROM a WHERE treatment = 'Control') c
ORDER BY incremental_spend_vs_control DESC;

-- name: q13_hillstrom_uplift_by_buyer_type
-- Q: Which e-mail works for which customer type? (conditional aggregation)
SELECT buyer_type,
       ROUND(AVG(CASE WHEN treatment = 'Mens' THEN conversion END)
           - AVG(CASE WHEN treatment = 'Control' THEN conversion END), 5) AS mens_conv_uplift,
       ROUND(AVG(CASE WHEN treatment = 'Womens' THEN conversion END)
           - AVG(CASE WHEN treatment = 'Control' THEN conversion END), 5) AS womens_conv_uplift,
       ROUND(AVG(CASE WHEN treatment = 'Mens' THEN spend END)
           - AVG(CASE WHEN treatment = 'Control' THEN spend END), 4) AS mens_spend_uplift,
       ROUND(AVG(CASE WHEN treatment = 'Womens' THEN spend END)
           - AVG(CASE WHEN treatment = 'Control' THEN spend END), 4) AS womens_spend_uplift,
       COUNT(*) AS customers
FROM hillstrom GROUP BY buyer_type ORDER BY customers DESC;

-- name: q14_targeting_priority_summary
-- Q: How many customers fall into each targeting priority, and what is the expected return?
SELECT t.priority, COUNT(*) AS customers,
       ROUND(AVG(t.p_next_campaign), 4) AS avg_p_next,
       ROUND(SUM(t.expected_profit_mu), 1) AS expected_profit_mu_if_contacted,
       ROUND(AVG(c.total_spend), 1) AS avg_hist_spend,
       SUM(CASE WHEN c.value_tier = 'High value' THEN 1 ELSE 0 END) AS high_value_customers
FROM targeting t JOIN customers c USING (ID)
GROUP BY t.priority ORDER BY t.priority;
