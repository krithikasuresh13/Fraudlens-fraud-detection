-- Reporting views (what Power BI reads). All carry run_id so a slicer can pick a run.

DROP VIEW IF EXISTS vw_transactions;
CREATE VIEW vw_transactions AS
SELECT run_id, txn_id, ts, DATE(ts) AS txn_date, CAST(STRFTIME('%H', ts) AS INTEGER) AS hour,
       customer_id, amount, merchant, category, channel, city, country, device_id,
       rule_score, ml_score, risk, tier, is_alert, case_id, is_fraud
FROM transactions;

DROP VIEW IF EXISTS vw_alerts;
CREATE VIEW vw_alerts AS
SELECT t.run_id, t.txn_id, t.ts, t.customer_id, t.amount, t.merchant, t.category, t.channel, t.city,
       t.risk, t.tier, t.case_id,
       (SELECT GROUP_CONCAT(r.rule_id, ', ') FROM alert_reasons r
         WHERE r.run_id = t.run_id AND r.txn_id = t.txn_id) AS rules_hit
FROM transactions t
WHERE t.is_alert = 1;

DROP VIEW IF EXISTS vw_case_queue;
CREATE VIEW vw_case_queue AS               -- analyst work queue with latest decision
SELECT c.run_id, c.case_id, c.customer_id, c.start_ts, c.end_ts, c.n_alerts, c.total_amount,
       c.case_risk, c.priority, c.rules, c.top_reason,
       COALESCE((SELECT a.action FROM case_actions a
                  WHERE a.run_id = c.run_id AND a.case_id = c.case_id
                  ORDER BY a.action_id DESC LIMIT 1), 'open') AS status
FROM cases c;

DROP VIEW IF EXISTS vw_rule_effectiveness;
CREATE VIEW vw_rule_effectiveness AS       -- how often each rule fires and (if labelled) how often it is right
SELECT r.run_id, r.rule_id,
       COUNT(*)                                   AS hits,
       ROUND(AVG(t.risk), 1)                      AS avg_risk,
       SUM(t.is_alert)                            AS alerts,
       SUM(COALESCE(t.is_fraud, 0))               AS fraud_hits,
       ROUND(AVG(t.is_fraud), 3)                  AS precision
FROM alert_reasons r
JOIN transactions t ON t.run_id = r.run_id AND t.txn_id = r.txn_id
GROUP BY r.run_id, r.rule_id;

DROP VIEW IF EXISTS vw_daily_summary;
CREATE VIEW vw_daily_summary AS
SELECT run_id, DATE(ts) AS txn_date,
       COUNT(*)                                        AS transactions,
       SUM(is_alert)                                   AS alerts,
       ROUND(100.0 * SUM(is_alert) / COUNT(*), 2)      AS alert_rate_pct,
       ROUND(SUM(CASE WHEN is_alert = 1 THEN amount ELSE 0 END), 2) AS amount_flagged
FROM transactions
GROUP BY run_id, DATE(ts);

DROP VIEW IF EXISTS vw_hourly_pattern;
CREATE VIEW vw_hourly_pattern AS           -- hour x weekday heat-map source
SELECT run_id,
       CAST(STRFTIME('%w', ts) AS INTEGER) AS weekday_no,       -- 0 = Sunday
       CAST(STRFTIME('%H', ts) AS INTEGER) AS hour,
       COUNT(*) AS transactions, SUM(is_alert) AS alerts
FROM transactions
GROUP BY run_id, weekday_no, hour;

DROP VIEW IF EXISTS vw_customer_risk;
CREATE VIEW vw_customer_risk AS
SELECT run_id, customer_id,
       COUNT(*) AS transactions, SUM(is_alert) AS alerts,
       ROUND(MAX(risk), 1) AS max_risk,
       ROUND(SUM(CASE WHEN is_alert = 1 THEN amount ELSE 0 END), 2) AS amount_flagged
FROM transactions
GROUP BY run_id, customer_id;

DROP VIEW IF EXISTS vw_linked_devices;
CREATE VIEW vw_linked_devices AS           -- devices shared by several customers (ring signal)
SELECT run_id, device_id,
       COUNT(DISTINCT customer_id) AS customers, COUNT(*) AS transactions,
       ROUND(SUM(amount), 2) AS total_amount, SUM(is_alert) AS alerts
FROM transactions
WHERE device_id <> 'unknown'
GROUP BY run_id, device_id
HAVING COUNT(DISTINCT customer_id) >= 3;
