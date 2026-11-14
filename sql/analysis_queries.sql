-- Ready-to-run analyst queries (SQLite). Replace :run with a run_id.

-- 1. Work queue: highest-priority open cases
SELECT case_id, customer_id, n_alerts, total_amount, case_risk, rules
FROM vw_case_queue WHERE run_id = :run AND status = 'open'
ORDER BY priority DESC LIMIT 10;

-- 2. Rule scorecard: which rules deserve their weight? (precision needs labelled data)
SELECT rule_id, hits, alerts, fraud_hits, precision
FROM vw_rule_effectiveness WHERE run_id = :run ORDER BY precision DESC, hits DESC;

-- 3. Seven-day moving average of daily alerts (window function) - spots alert spikes
SELECT txn_date, alerts,
       ROUND(AVG(alerts) OVER (ORDER BY txn_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 1) AS alerts_7d_avg
FROM vw_daily_summary WHERE run_id = :run ORDER BY txn_date;

-- 4. Repeat offenders: customers with more than one case
SELECT customer_id, COUNT(*) AS cases, ROUND(SUM(total_amount), 0) AS exposure
FROM cases WHERE run_id = :run
GROUP BY customer_id HAVING COUNT(*) > 1 ORDER BY exposure DESC;

-- 5. Ring hunting: devices shared by several customers, with who used them
SELECT d.device_id, d.customers, d.total_amount,
       GROUP_CONCAT(DISTINCT t.customer_id) AS customer_list
FROM vw_linked_devices d
JOIN transactions t ON t.run_id = d.run_id AND t.device_id = d.device_id
WHERE d.run_id = :run
GROUP BY d.device_id ORDER BY d.customers DESC;

-- 6. Peak risk hours: share of transactions that alert, by hour of day
SELECT hour, SUM(transactions) AS txns, SUM(alerts) AS alerts,
       ROUND(100.0 * SUM(alerts) / SUM(transactions), 2) AS alert_rate_pct
FROM vw_hourly_pattern WHERE run_id = :run GROUP BY hour ORDER BY alert_rate_pct DESC LIMIT 5;

-- 7. Analyst feedback loop: false-positive rate by rule (needs decisions in case_actions)
SELECT r.rule_id,
       COUNT(DISTINCT a.case_id)                                              AS reviewed_cases,
       SUM(a.action = 'false_positive')                                       AS false_positives,
       ROUND(100.0 * SUM(a.action = 'false_positive') / COUNT(*), 1)          AS fp_rate_pct
FROM case_actions a
JOIN transactions t ON t.run_id = a.run_id AND t.case_id = a.case_id AND t.is_alert = 1
JOIN alert_reasons r ON r.run_id = t.run_id AND r.txn_id = t.txn_id
WHERE a.run_id = :run
GROUP BY r.rule_id ORDER BY fp_rate_pct DESC;
