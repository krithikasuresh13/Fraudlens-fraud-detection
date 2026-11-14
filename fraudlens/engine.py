"""Detection engine: customer baselines -> rules -> Isolation Forest -> blended risk -> cases."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED = ["timestamp", "customer_id", "amount"]
SMALL_AMT = 25.0

# weight = how much a hit alone raises the probability-like risk (combined with noisy-OR)
RULES = {
    "AMOUNT_SPIKE":      dict(weight=0.35, title="Amount spike", desc="Amount is far above this customer's own normal spend (robust z-score >= 4). Weighted below the alert threshold on purpose: big purchases are often legitimate, so it needs a second signal."),
    "VELOCITY_BURST":    dict(weight=0.55, title="Velocity burst", desc="5+ transactions by one customer inside 10 minutes."),
    "IMPOSSIBLE_TRAVEL": dict(weight=0.75, title="Impossible travel", desc="Two transactions too far apart to travel between (>=300 km at >=800 km/h)."),
    "CARD_TEST_BURST":   dict(weight=0.50, title="Card-testing burst", desc="3+ tiny payments (<= 25) inside 15 minutes."),
    "CARD_TEST_CASHOUT": dict(weight=0.70, title="Card-test cash-out", desc="A larger payment within 30 minutes after a card-testing burst."),
    "NEW_DEVICE_HIGH":   dict(weight=0.45, title="New device + high amount", desc="First use of a device by an established customer, with an unusually high amount."),
    "ODD_HOUR_HIGH":     dict(weight=0.35, title="Odd-hour high value", desc="Large payment between 00:00-04:59 from a customer who almost never transacts at night."),
    "SHARED_DEVICE":     dict(weight=0.50, title="Shared device (possible ring)", desc="One device is used by 4+ different customers."),
    "FIRST_FOREIGN":     dict(weight=0.35, title="First foreign transaction", desc="First ever transaction outside the customer's home country."),
}
DEFAULT_WEIGHTS = {k: v["weight"] for k, v in RULES.items()}


def _m(x: float) -> str:
    return f"₹{x:,.0f}"


TEXT = {
    "AMOUNT_SPIKE": lambda r: f"{_m(r.amount)} is {r.amount / max(r.cust_median, 1):.1f}x this customer's typical {_m(r.cust_median)} (z={r.z_amt:.1f})",
    "VELOCITY_BURST": lambda r: f"{int(r.vel10)} transactions within 10 minutes",
    "IMPOSSIBLE_TRAVEL": lambda r: f"{r.dist_prev:,.0f} km from the previous transaction after {r.gap_min:.0f} min (~{r.speed:,.0f} km/h)",
    "CARD_TEST_BURST": lambda r: f"burst of 3+ tiny payments (<= {_m(SMALL_AMT)}) within 15 minutes",
    "CARD_TEST_CASHOUT": lambda r: f"{_m(r.amount)} shortly after a card-testing burst",
    "NEW_DEVICE_HIGH": lambda r: f"first use of device {r.device_id} with an unusually high amount (z={r.z_amt:.1f})",
    "ODD_HOUR_HIGH": lambda r: f"{r.hour:02d}:xx - customer rarely transacts at night; amount z={r.z_amt:.1f}",
    "SHARED_DEVICE": lambda r: f"device {r.device_id} is used by {int(r.device_customers)} different customers",
    "FIRST_FOREIGN": lambda r: f"first transaction ever in {r.country} (home country {r.home_country})",
}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ preparation

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Need at least {REQUIRED}.")
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    d = d.dropna(subset=REQUIRED)
    d = d[d["amount"] >= 0]
    if "txn_id" not in d.columns:
        d["txn_id"] = [f"T{i:07d}" for i in range(len(d))]
    for col in ("merchant", "category", "channel", "city", "country", "device_id"):
        d[col] = d[col].fillna("unknown").astype(str) if col in d.columns else "unknown"
    for col in ("lat", "lon"):
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d.columns else np.nan
    d["customer_id"] = d["customer_id"].astype(str)
    return d.sort_values(["customer_id", "timestamp"], kind="stable").reset_index(drop=True)


def _win(t: np.ndarray, seconds: float):
    """Counts of events within `seconds` behind / ahead of each event (inclusive of itself)."""
    i = np.arange(len(t))
    back = i - np.searchsorted(t, t - seconds, "left") + 1
    fwd = np.searchsorted(t, t + seconds, "right") - i
    return back, fwd


def compute_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["t_sec"] = (d["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    cust = d["customer_id"]

    # --- per-customer robust amount baseline (median / MAD on log scale)
    d["log_amt"] = np.log1p(d["amount"])
    g = d.groupby("customer_id")["log_amt"]
    cnt, med = g.transform("count"), g.transform("median")
    mad = (d["log_amt"] - med).abs().groupby(cust).transform("median")
    gmed = d["log_amt"].median()
    gmad = (d["log_amt"] - gmed).abs().median()
    thin = cnt < 8                                      # too little history -> use population baseline
    med = med.where(~thin, gmed)
    scale = (1.4826 * mad).where(~thin, 1.4826 * gmad).clip(lower=0.25)
    d["z_amt"] = (d["log_amt"] - med) / scale
    d["cust_median"] = np.expm1(med)

    d["hour"] = d["timestamp"].dt.hour
    d["night_share"] = (d["hour"] < 5).astype(float).groupby(cust).transform("mean")

    # --- windowed behaviour per customer
    n = len(d)
    vel10, tx1h = np.zeros(n), np.zeros(n)
    ctb, cto = np.zeros(n, bool), np.zeros(n, bool)
    ts, amt = d["t_sec"].values, d["amount"].values
    for _cid, idx in d.groupby("customer_id").indices.items():
        t, a = ts[idx], amt[idx]
        b10, f10 = _win(t, 600)
        vel10[idx] = np.maximum(b10, f10)
        tx1h[idx] = _win(t, 3600)[0]
        sm = a <= SMALL_AMT
        if sm.sum() >= 3:
            t_s = t[sm]
            bs, fs = _win(t_s, 900)
            flag = np.maximum(bs, fs) >= 3
            pos = np.where(sm)[0]
            ctb[idx[pos[flag]]] = True
            fl = t_s[flag]
            if len(fl):
                before = np.searchsorted(fl, t, "left") - np.searchsorted(fl, t - 1800, "left")
                cto[idx[(~sm) & (a >= 500) & (before > 0)]] = True
    d["vel10"], d["tx1h"], d["ct_burst"], d["ct_cashout"] = vel10, tx1h, ctb, cto

    # --- geography
    gc = d.groupby("customer_id")
    d["gap_min"] = (d["t_sec"] - gc["t_sec"].shift()) / 60
    d["dist_prev"] = haversine(gc["lat"].shift(), gc["lon"].shift(), d["lat"], d["lon"])
    d["speed"] = d["dist_prev"] / (d["gap_min"] / 60).clip(lower=1 / 60)
    d["dist_home"] = pd.Series(
        haversine(gc["lat"].transform("median"), gc["lon"].transform("median"), d["lat"], d["lon"])).fillna(0.0)

    # --- devices
    known = d["device_id"].ne("unknown")
    first_seen = d.groupby(["customer_id", "device_id"])["timestamp"].transform("min")
    cust_first = gc["timestamp"].transform("min")
    d["is_new_device"] = known & ((d["timestamp"] - first_seen) < pd.Timedelta(hours=24)) \
        & ((first_seen - cust_first) > pd.Timedelta(days=7))
    d["device_customers"] = d.groupby("device_id")["customer_id"].transform("nunique").where(known, 0)

    # --- country
    home = d.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0])
    d["home_country"] = d["customer_id"].map(home)
    foreign = d["country"].ne("unknown") & d["country"].ne(d["home_country"])
    d["is_foreign"] = foreign
    d["first_foreign"] = foreign & (d.groupby(["customer_id", "country"]).cumcount() == 0)
    return d


# -------------------------------------------------------------------- rules

def apply_rules(d: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    d = d.copy()
    big = d["amount"] >= 1000
    d["r_AMOUNT_SPIKE"] = (d["z_amt"] >= 4) & big
    d["r_VELOCITY_BURST"] = d["vel10"] >= 5
    d["r_IMPOSSIBLE_TRAVEL"] = (d["dist_prev"] >= 300) & (d["speed"] >= 800)
    d["r_CARD_TEST_BURST"] = d["ct_burst"]
    d["r_CARD_TEST_CASHOUT"] = d["ct_cashout"]
    d["r_NEW_DEVICE_HIGH"] = d["is_new_device"] & (d["z_amt"] >= 2.5) & big
    d["r_ODD_HOUR_HIGH"] = (d["hour"] < 5) & (d["night_share"] < 0.05) & (d["z_amt"] >= 2) & big
    d["r_SHARED_DEVICE"] = d["device_customers"] >= 4
    d["r_FIRST_FOREIGN"] = d["first_foreign"]

    keep = np.ones(len(d))
    for rid in RULES:
        keep *= np.where(d[f"r_{rid}"], 1 - w[rid], 1.0)
    d["rule_score"] = 100 * (1 - keep)

    reasons = [[] for _ in range(len(d))]
    hit = d[[f"r_{r}" for r in RULES]].any(axis=1)
    for row in d[hit].itertuples():
        for rid in RULES:
            if getattr(row, f"r_{rid}"):
                reasons[row.Index].append((rid, TEXT[rid](row)))
    d["reasons"] = reasons
    return d


# ----------------------------------------------------------------------- ML

ML_LABELS = {"log_amt": "amount", "z_amt": "amount vs. customer norm", "hour_sin": "time of day", "hour_cos": "time of day",
             "log_gap": "time since last transaction", "tx1h": "transaction frequency", "dist_home": "distance from home",
             "is_new_device": "new device", "is_foreign": "foreign location"}


def apply_ml(d: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    d = d.copy()
    ang = 2 * np.pi * d["hour"] / 24
    X = pd.DataFrame({
        "log_amt": d["log_amt"], "z_amt": d["z_amt"], "hour_sin": np.sin(ang), "hour_cos": np.cos(ang),
        "log_gap": np.log1p(d["gap_min"].fillna(10080).clip(lower=0)), "tx1h": d["tx1h"],
        "dist_home": np.log1p(d["dist_home"]), "is_new_device": d["is_new_device"].astype(float),
        "is_foreign": d["is_foreign"].astype(float),
    }).join(pd.get_dummies(d["channel"], prefix="ch").astype(float))
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X)
    raw = -model.score_samples(X)
    pct = pd.Series(raw).rank(pct=True).values
    d["ml_pct"] = pct
    d["ml_score"] = 100 * np.clip((pct - 0.97) / 0.03, 0, 1)     # only the top 3% carry ML signal

    # explain: which features sit furthest from the population for each flagged row
    z = ((X - X.mean()) / X.std().replace(0, 1)).fillna(0)
    feat_cols = [c for c in X.columns if c in ML_LABELS]
    flagged = np.where(d["ml_score"].values > 0)[0]
    why = {}
    for i in flagged:
        zi = z.iloc[i][feat_cols]
        names = []
        for f in zi.abs().sort_values(ascending=False).index:
            lab = f"{ML_LABELS[f]} ({'high' if zi[f] > 0 else 'low'})"
            if lab.split(" (")[0] not in [n.split(" (")[0] for n in names]:
                names.append(lab)
            if len(names) == 2:
                break
        why[i] = ("ML_ANOMALY", f"statistically unusual vs. all transactions: {', '.join(names)} (top {100 * (1 - pct[i]):.1f}%)")
    d["reasons"] = [r + ([why[i]] if i in why else []) for i, r in enumerate(d["reasons"])]
    return d


def tier(risk: float) -> str:
    return "High" if risk >= 70 else "Medium" if risk >= 40 else "Low" if risk >= 20 else "Normal"


# ------------------------------------------------------------------- pipeline

@dataclass
class Result:
    df: pd.DataFrame
    alerts: pd.DataFrame
    cases: pd.DataFrame
    rule_summary: pd.DataFrame
    sweep: pd.DataFrame
    devices: pd.DataFrame
    metrics: dict
    threshold: float
    labeled: bool
    weights: dict = field(default_factory=dict)


def reasons_text(rs) -> str:
    return " | ".join(f"{rid}: {t}" for rid, t in rs)


def build_cases(alerts: pd.DataFrame, gap_min: float) -> tuple[pd.DataFrame, pd.Series]:
    if alerts.empty:
        return pd.DataFrame(columns=["case_id", "customer_id", "start", "end", "n_alerts", "total_amount", "max_risk",
                                     "case_risk", "priority", "rules", "top_reason"]), pd.Series(dtype=object)
    a = alerts.sort_values(["customer_id", "timestamp"]).copy()
    new = (a["customer_id"] != a["customer_id"].shift()) | \
          ((a["timestamp"] - a["timestamp"].shift()).dt.total_seconds() > gap_min * 60)
    a["case_no"] = new.cumsum()
    rows = []
    for no, grp in a.groupby("case_no"):
        top = grp.loc[grp["risk"].idxmax()]
        rules = sorted({rid for rs in grp["reasons"] for rid, _ in rs})
        total = grp["amount"].sum()
        case_risk = min(100.0, grp["risk"].max() + 3 * (len(grp) - 1))
        rows.append(dict(case_no=no, customer_id=grp["customer_id"].iat[0], start=grp["timestamp"].min(),
                         end=grp["timestamp"].max(), n_alerts=len(grp), total_amount=round(total, 2),
                         max_risk=grp["risk"].max(), case_risk=round(case_risk, 1),
                         priority=round(case_risk * np.log10(total + 10), 1), rules=", ".join(rules),
                         top_reason=top["reasons"][0][1] if top["reasons"] else "",
                         has_fraud=int(grp["is_fraud"].fillna(0).max()) if "is_fraud" in grp else None))
    c = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    c.insert(0, "case_id", [f"CASE-{i:04d}" for i in range(1, len(c) + 1)])
    mapping = pd.Series(c["case_id"].values, index=c["case_no"].values)
    return c.drop(columns=["case_no"]), a["case_no"].map(mapping)


def run_pipeline(raw: pd.DataFrame, weights: dict | None = None, threshold: float = 40.0,
                 gap_min: float = 120, ml_weight: float = 0.5, seed: int = 7) -> Result:
    d = apply_ml(apply_rules(compute_features(prepare(raw)), weights), seed)
    p = 1 - (1 - d["rule_score"] / 100) * (1 - ml_weight * d["ml_score"] / 100)
    d["risk"] = (100 * p).round(1)
    d["tier"] = d["risk"].map(tier)
    d["is_alert"] = d["risk"] >= threshold
    labeled = "is_fraud" in d.columns
    if labeled:
        d["is_fraud"] = pd.to_numeric(d["is_fraud"], errors="coerce").fillna(0).astype(int)

    alerts = d[d["is_alert"]].copy()
    cases, case_map = build_cases(alerts, gap_min)
    d["case_id"] = case_map.reindex(d.index) if len(case_map) else None
    # case_map is indexed by alert row index (kept from alerts.sort_values), so reindex lines up
    alerts["case_id"] = d.loc[alerts.index, "case_id"]
    d = d.sort_values("timestamp").reset_index(drop=True)
    alerts = alerts.sort_values("risk", ascending=False)

    days = max((d["timestamp"].max() - d["timestamp"].min()).days + 1, 1)
    sweep_rows = []
    for thr in range(10, 95, 5):
        sel = d["risk"] >= thr
        row = {"threshold": thr, "alerts": int(sel.sum()), "alerts_per_day": round(sel.sum() / days, 1)}
        if labeled:
            tp = int((sel & (d["is_fraud"] == 1)).sum())
            row["precision"] = round(tp / sel.sum(), 3) if sel.sum() else np.nan
            row["recall"] = round(tp / max(int(d["is_fraud"].sum()), 1), 3)
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)

    rs = []
    for rid, meta in RULES.items():
        hit = d[f"r_{rid}"]
        row = {"rule": rid, "title": meta["title"], "weight": (weights or {}).get(rid, meta["weight"]),
               "hits": int(hit.sum()), "avg_risk": round(d.loc[hit, "risk"].mean(), 1) if hit.any() else np.nan}
        if labeled:
            row["precision"] = round(d.loc[hit, "is_fraud"].mean(), 3) if hit.any() else np.nan
            row["fraud_caught"] = int(d.loc[hit, "is_fraud"].sum())
        rs.append(row)
    ml_hit = d["ml_score"] > 0
    row = {"rule": "ML_ANOMALY", "title": "Isolation Forest (top 3%)", "weight": ml_weight, "hits": int(ml_hit.sum()),
           "avg_risk": round(d.loc[ml_hit, "risk"].mean(), 1) if ml_hit.any() else np.nan}
    if labeled:
        row["precision"] = round(d.loc[ml_hit, "is_fraud"].mean(), 3) if ml_hit.any() else np.nan
        row["fraud_caught"] = int(d.loc[ml_hit, "is_fraud"].sum())
    rs.append(row)
    rule_summary = pd.DataFrame(rs)

    known = d[d["device_id"] != "unknown"]
    dev = known.groupby("device_id").agg(customers=("customer_id", "nunique"), txns=("txn_id", "count"),
                                         total_amount=("amount", "sum"), alerts=("is_alert", "sum")).reset_index()
    devices = dev[dev["customers"] >= 3].sort_values(["customers", "total_amount"], ascending=False).reset_index(drop=True)

    metrics = {"transactions": len(d), "customers": d["customer_id"].nunique(), "alerts": int(d["is_alert"].sum()),
               "alert_rate": round(100 * d["is_alert"].mean(), 2), "cases": len(cases), "days": days,
               "amount_flagged": float(alerts["amount"].sum()), "high": int((d["tier"] == "High").sum())}
    if labeled:
        tp = int((d["is_alert"] & (d["is_fraud"] == 1)).sum())
        nf = int(d["is_fraud"].sum())
        prec, rec = tp / max(metrics["alerts"], 1), tp / max(nf, 1)
        metrics.update(fraud_txns=nf, tp=tp, precision=round(prec, 3), recall=round(rec, 3),
                       f1=round(2 * prec * rec / max(prec + rec, 1e-9), 3))
        if len(cases):
            metrics["case_precision"] = round(float(cases["has_fraud"].mean()), 3)
    return Result(d, alerts, cases, rule_summary, sweep, devices, metrics, threshold, labeled, weights or {})
