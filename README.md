#  FraudLens — explainable suspicious-transaction screening

Finds unusual financial transactions, explains *why* each one is suspicious, groups alerts into cases, and ranks the cases
by **risk × money at stake** so analysts work the right thing first.
Stack: Python · SQL (SQLite) · Streamlit · Power BI.

## How it works

```
transactions ─► per-customer baseline ─► 9 transparent rules ─┐
                (robust median/MAD)                            ├─► blended risk 0-100 ─► alerts ─► cases ─► work queue
              ─► Isolation Forest (top 3% unusual) ───────────┘
```

| Idea | Detail |
|---|---|
| **"Unusual *for this customer*"** | Amounts are scored against each customer's own robust baseline (median / MAD on log scale), not a global limit. ₹20,000 is routine for one person and alarming for another. |
| **9 explainable rules** | Amount spike · velocity burst · impossible travel · card-testing burst · card-test cash-out · new device + high amount · odd-hour high value · shared device (ring) · first foreign transaction. |
| **Reason codes in plain English** | e.g. `₹48,200 is 14.2x this customer's typical ₹3,400` / `612 km from the previous transaction after 25 min (~1,450 km/h)`. |
| **Signals combine** | Each hit has a weight; hits merge as `1 − Π(1 − w)`. A big amount alone stays *below* the alert line (big purchases are often legitimate); a big amount **plus** a new device crosses it. |
| **Anomaly model as a second opinion** | Isolation Forest on amount, timing, frequency, distance, device and channel. Only the top 3% most unusual transactions add risk, and the top two driving features are named. |
| **Cases, not a flood of alerts** | Alerts for one customer within a window (default 2 h) become one case with total exposure; the queue is ordered by `case_risk × log10(exposure)`. |
| **Alert-budget tuner** | See alerts/day, precision and recall at every threshold, so you pick what your team can staff. |
| **Analyst feedback loop** | Decisions (confirm / false positive / escalate / monitor) are stored as an audit trail; a SQL query turns them into false-positive rate per rule. |
| **Ring spotting** | Devices shared by several customers are surfaced separately. |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py                  # analyst console
python run_demo.py                    # same pipeline in the terminal -> fraudlens.db + powerbi/data/*.csv
python run_demo.py --csv my_transactions.csv --threshold 45
python -m unittest discover tests
```

**Your own data:** CSV with `timestamp`, `customer_id`, `amount` (required). Optional columns unlock more rules:
`txn_id, merchant, category, channel, city, country, lat, lon` (impossible travel), `device_id` (new device / rings),
`is_fraud` (0/1 — only used to *report* accuracy, never to score).

## About the demo data (please read)

No dataset was provided, so `fraudlens/datagen.py` creates ~18k transactions for 300 customers over 90 days, with benign
oddities (occasional big purchases, new phones, travel) and **seven planted fraud patterns** (148 fraudulent transactions).
At the default threshold the demo reports roughly **89% precision / 95% recall**. Treat those as a **sanity check, not a
benchmark**: the rules were written for the same patterns the generator plants, and I set the default `AMOUNT_SPIKE`
weight (0.35) by looking at this data. Real fraud is messier — expect lower numbers and re-tune on your own history.

## SQL layer

`sql/schema.sql` runs, transactions, alert_reasons, cases, case_actions · `sql/views.sql` 8 reporting views ·
`sql/analysis_queries.sql` work queue, rule scorecard, 7-day moving average (window function), repeat offenders,
ring hunting, peak-risk hours, analyst false-positive rate per rule.

## Power BI (5 minutes)

1. In the app's **Export** tab click *Save run to SQLite + build Power BI pack* (or run `run_demo.py`) and unzip.
2. Power BI Desktop → **Get data → Text/CSV** → load the `vw_*.csv` files (keep one run in the model, or slicer on `run_id`).
3. Relate `vw_transactions[case_id]` → `vw_case_queue[case_id]`.
4. Add `powerbi/measures.dax`.
5. Suggested page: KPI cards (Transactions, Alerts, Alert Rate %, Amount Flagged, Open Cases) · column chart of `vw_daily_summary` alerts by `txn_date` ·
   matrix `vw_hourly_pattern` weekday × hour with `Hourly Alert Rate` as heat-map · bar chart of `vw_rule_effectiveness` hits by `rule_id` ·
   table from `vw_case_queue` sorted by `priority` · slicer on `tier`.

## Known limits (before using on real money)

* **Baselines use each customer's whole history** (batch screening). A live system must use *prior-only* rolling baselines to avoid look-ahead.
* Customers with < 8 transactions fall back to the population baseline, so new accounts are scored more coarsely.
* Rules are thresholds I chose (`engine.py`, `RULES` and `apply_rules`); validate them against your confirmed-fraud history.
* The Isolation Forest is unsupervised. With labelled history, add a supervised model (e.g. gradient boosting) and compare.
* This is decision support for analysts, not an automatic block/decline engine.
