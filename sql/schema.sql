-- FraudLens schema (SQLite). Every screening run is stored under a run_id so runs can be compared.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    source          TEXT,
    threshold       REAL,
    n_transactions  INTEGER,
    n_alerts        INTEGER,
    n_cases         INTEGER
);

CREATE TABLE IF NOT EXISTS transactions (
    run_id       INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    txn_id       TEXT NOT NULL,
    ts           TEXT NOT NULL,                -- 'YYYY-MM-DD HH:MM:SS'
    customer_id  TEXT NOT NULL,
    amount       REAL NOT NULL,
    merchant     TEXT, category TEXT, channel TEXT, city TEXT, country TEXT, device_id TEXT,
    rule_score   REAL, ml_score REAL, risk REAL, tier TEXT,
    is_alert     INTEGER NOT NULL DEFAULT 0,
    case_id      TEXT,
    is_fraud     INTEGER,                      -- ground truth, only present for labelled data
    PRIMARY KEY (run_id, txn_id)
);

CREATE TABLE IF NOT EXISTS alert_reasons (    -- why each transaction was scored the way it was
    run_id   INTEGER NOT NULL,
    txn_id   TEXT NOT NULL,
    rule_id  TEXT NOT NULL,
    detail   TEXT,
    FOREIGN KEY (run_id, txn_id) REFERENCES transactions(run_id, txn_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cases (            -- alerts for one customer close in time are grouped
    run_id        INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    case_id       TEXT NOT NULL,
    customer_id   TEXT NOT NULL,
    start_ts      TEXT, end_ts TEXT,
    n_alerts      INTEGER,
    total_amount  REAL,
    case_risk     REAL,
    priority      REAL,                         -- case_risk x log10(exposure): what analysts work first
    rules         TEXT,
    top_reason    TEXT,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS case_actions (     -- analyst decisions (append-only audit trail)
    action_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL,
    case_id    TEXT NOT NULL,
    action     TEXT NOT NULL CHECK (action IN ('confirm_fraud', 'false_positive', 'escalate', 'monitor')),
    note       TEXT,
    acted_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id, case_id) REFERENCES cases(run_id, case_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_txn_customer ON transactions(run_id, customer_id, ts);
CREATE INDEX IF NOT EXISTS ix_txn_alert    ON transactions(run_id, is_alert, risk DESC);
CREATE INDEX IF NOT EXISTS ix_reason_rule  ON alert_reasons(run_id, rule_id);
