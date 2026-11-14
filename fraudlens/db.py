"""SQLite persistence and Power BI export."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .engine import Result

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
VIEWS = ["vw_transactions", "vw_alerts", "vw_case_queue", "vw_rule_effectiveness", "vw_daily_summary",
         "vw_hourly_pattern", "vw_customer_risk", "vw_linked_devices"]


def connect(path: str | Path = "fraudlens.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    conn.executescript((SQL_DIR / "views.sql").read_text())
    conn.commit()
    return conn


def save_run(conn: sqlite3.Connection, res: Result, source: str = "app", actions: list[tuple] | None = None) -> int:
    """Persist a run. `actions` = [(case_id, action, note), ...] from analyst decisions."""
    m = res.metrics
    cur = conn.execute("INSERT INTO runs(source, threshold, n_transactions, n_alerts, n_cases) VALUES (?,?,?,?,?)",
                       (source, res.threshold, m["transactions"], m["alerts"], m["cases"]))
    run_id = cur.lastrowid
    d = res.df
    t = pd.DataFrame({
        "run_id": run_id, "txn_id": d["txn_id"], "ts": d["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id": d["customer_id"], "amount": d["amount"], "merchant": d["merchant"], "category": d["category"],
        "channel": d["channel"], "city": d["city"], "country": d["country"], "device_id": d["device_id"],
        "rule_score": d["rule_score"].round(1), "ml_score": d["ml_score"].round(1), "risk": d["risk"], "tier": d["tier"],
        "is_alert": d["is_alert"].astype(int), "case_id": d["case_id"],
        "is_fraud": d["is_fraud"] if res.labeled else None})
    t.to_sql("transactions", conn, if_exists="append", index=False)

    rr = [(run_id, tid, rid, txt) for tid, rs in zip(d["txn_id"], d["reasons"]) for rid, txt in rs]
    conn.executemany("INSERT INTO alert_reasons VALUES (?,?,?,?)", rr)

    c = res.cases
    if len(c):
        conn.executemany(
            "INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(run_id, r.case_id, r.customer_id, r.start.strftime("%Y-%m-%d %H:%M:%S"), r.end.strftime("%Y-%m-%d %H:%M:%S"),
              int(r.n_alerts), float(r.total_amount), float(r.case_risk), float(r.priority), r.rules, r.top_reason)
             for r in c.itertuples()])
    valid = set(c["case_id"]) if len(c) else set()
    conn.executemany("INSERT INTO case_actions(run_id, case_id, action, note) VALUES (?,?,?,?)",
                     [(run_id, cid, act, note) for cid, act, note in (actions or []) if cid in valid])
    conn.commit()
    return run_id


def export_powerbi(conn: sqlite3.Connection, out_dir: str | Path, run_id: int | None = None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for v in VIEWS:
        q = f"SELECT * FROM {v}" + (f" WHERE run_id = {int(run_id)}" if run_id else "")
        p = out / f"{v}.csv"
        pd.read_sql_query(q, conn).round(3).to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(p)
    return paths
