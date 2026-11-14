"""FraudLens - analyst console.   Run:  streamlit run app.py"""
import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from fraudlens.datagen import generate
from fraudlens.db import connect, export_powerbi, save_run
from fraudlens.engine import DEFAULT_WEIGHTS, RULES, reasons_text, run_pipeline

HERE = Path(__file__).parent
ACTIONS = {"Confirm fraud": "confirm_fraud", "False positive": "false_positive", "Escalate": "escalate", "Keep monitoring": "monitor"}

st.set_page_config(page_title="FraudLens - suspicious transaction screening", page_icon="🔎", layout="wide")
st.title("🔎 FraudLens")
st.caption("Explainable fraud screening: per-customer behaviour baselines + transparent rules + an anomaly model, "
           "with alerts grouped into cases and ranked by risk × money at stake.")


@st.cache_data(show_spinner="Generating synthetic transactions…")
def load_synth(n_customers, days, seed):
    return generate(n_customers, days, seed)


@st.cache_data(show_spinner="Scoring transactions…")
def run(raw, weights_items, threshold, gap_min, ml_weight):
    return run_pipeline(raw, dict(weights_items), threshold, gap_min, ml_weight)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Data")
    src = st.radio("Source", ["Synthetic demo", "Upload CSV"])
    if src == "Synthetic demo":
        n_c = st.slider("Customers", 100, 1000, 300, 50)
        days = st.slider("Days", 30, 120, 90, 10)
        seed = st.number_input("Random seed", 0, 9999, 42)
        raw = load_synth(n_c, days, int(seed))
        source_name = "synthetic"
    else:
        up = st.file_uploader("Transactions CSV", type=["csv"])
        with st.expander("Expected columns"):
            st.markdown("**Required:** `timestamp`, `customer_id`, `amount`  \n"
                        "**Optional (unlock more rules):** `txn_id`, `merchant`, `category`, `channel`, `city`, `country`, "
                        "`lat`, `lon` (impossible travel), `device_id` (new device / rings), `is_fraud` (0/1, enables accuracy metrics)")
        if up is None:
            st.info("Upload a CSV, or switch to the synthetic demo.")
            st.stop()
        raw, source_name = pd.read_csv(up), up.name

    st.header("2 · Sensitivity")
    threshold = st.slider("Alert when risk ≥", 10, 90, 40)
    gap_min = st.slider("Group alerts into one case if within (min)", 15, 720, 120, 15)
    ml_w = st.slider("Anomaly-model influence", 0, 100, 50) / 100
    with st.expander("Rule weights (how much one hit raises risk)"):
        weights = {rid: st.slider(RULES[rid]["title"], 0, 95, int(DEFAULT_WEIGHTS[rid] * 100), key=f"w_{rid}") / 100
                   for rid in RULES}

try:
    res = run(raw, tuple(sorted(weights.items())), threshold, gap_min, ml_w)
except ValueError as e:
    st.error(str(e))
    st.stop()
m, d = res.metrics, res.df

tabs = st.tabs(["📊 Overview", "🗂️ Case queue", "🧪 Rule lab", "🎚️ Threshold tuner", "🕸️ Linked accounts", "📤 Export / Power BI"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    k = st.columns(5)
    k[0].metric("Transactions", f"{m['transactions']:,}")
    k[1].metric("Alerts", f"{m['alerts']:,}", f"{m['alert_rate']}% of volume", delta_color="off")
    k[2].metric("Cases to review", f"{m['cases']:,}")
    k[3].metric("Amount flagged", f"₹{m['amount_flagged']:,.0f}")
    k[4].metric("High-risk txns", f"{m['high']:,}")
    if res.labeled:
        st.success(f"Against the `is_fraud` labels at this threshold: precision **{m['precision']:.1%}**, recall **{m['recall']:.1%}**, "
                   f"F1 **{m['f1']:.2f}**" + (f", case precision **{m['case_precision']:.1%}**." if "case_precision" in m else "."))
    a, b = st.columns(2)
    with a:
        st.markdown("**Alerts per day**")
        st.bar_chart(d.groupby(d["timestamp"].dt.date)["is_alert"].sum())
    with b:
        st.markdown("**Alert rate by hour of day** (% of transactions that alert)")
        st.bar_chart(d.groupby("hour")["is_alert"].mean().mul(100).round(2))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Risk tiers**")
        st.bar_chart(d["tier"].value_counts().reindex(["High", "Medium", "Low", "Normal"]).fillna(0))
    with c2:
        st.markdown("**Alerts by channel**")
        st.bar_chart(d[d["is_alert"]].groupby("channel").size())
    with c3:
        st.markdown("**Alerts by category**")
        st.bar_chart(d[d["is_alert"]].groupby("category").size())
    st.markdown("**Highest-risk transactions**")
    top = res.alerts.head(15).copy()
    top["why"] = top["reasons"].map(reasons_text)
    st.dataframe(top[["txn_id", "timestamp", "customer_id", "amount", "merchant", "channel", "city", "risk", "tier", "why"]],
                 hide_index=True, use_container_width=True,
                 column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%.0f"),
                                "why": st.column_config.TextColumn("Why flagged", width="large")})

# ------------------------------------------------------------------ cases
with tabs[1]:
    cases = res.cases
    if cases.empty:
        st.info("No alerts at this threshold. Lower it in the sidebar.")
    else:
        st.session_state.setdefault("decisions", {})
        dec = st.session_state["decisions"]
        key = lambda r: f"{r.customer_id}|{r.start}"
        view = cases.copy()
        view["status"] = [dec.get(key(r), ("open", ""))[0] for r in cases.itertuples()]
        st.markdown("**Work queue** - ordered by *case risk × log(money at stake)*")
        st.dataframe(view[["case_id", "customer_id", "start", "n_alerts", "total_amount", "case_risk", "rules", "status"]],
                     hide_index=True, use_container_width=True,
                     column_config={"case_risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%.0f")})
        pick = st.selectbox("Open a case", view["case_id"])
        c = cases[cases["case_id"] == pick].iloc[0]
        st.subheader(f"{pick} · customer {c.customer_id} · ₹{c.total_amount:,.0f} across {c.n_alerts} alert(s)")
        ca = res.alerts[res.alerts["case_id"] == pick].sort_values("timestamp")
        for r in ca.itertuples():
            with st.expander(f"{r.timestamp:%d %b %H:%M} · ₹{r.amount:,.0f} · {r.merchant} · risk {r.risk:.0f}", expanded=len(ca) <= 3):
                for rid, txt in r.reasons:
                    st.markdown(f"- **{rid}** - {txt}")
        st.markdown("**Customer timeline** (±6 hours around the case)")
        win = d[(d["customer_id"] == c.customer_id) & (d["timestamp"] >= c.start - pd.Timedelta(hours=6))
                & (d["timestamp"] <= c.end + pd.Timedelta(hours=6))]
        st.dataframe(win[["timestamp", "amount", "merchant", "channel", "city", "device_id", "risk", "is_alert"]],
                     hide_index=True, use_container_width=True,
                     column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%.0f")})
        st.markdown("**Analyst decision**")
        d1, d2 = st.columns([1, 2])
        choice = d1.radio("Outcome", list(ACTIONS), key=f"act_{pick}")
        note = d2.text_input("Note", key=f"note_{pick}")
        if st.button("Record decision", key=f"rec_{pick}"):
            dec[key(c)] = (ACTIONS[choice], note)
            st.success("Recorded. Save the run in the Export tab to persist decisions to SQL.")

# ------------------------------------------------------------------ rule lab
with tabs[2]:
    st.markdown("**Every rule, how often it fires, and (with labels) how often it is right.**")
    rs = res.rule_summary.copy()
    rs["what it checks"] = rs["rule"].map(lambda r: RULES[r]["desc"] if r in RULES else
                                          "Isolation Forest over amount, timing, frequency, distance, device and channel; only the top 3% most unusual transactions add risk.")
    cfg = {"what it checks": st.column_config.TextColumn(width="large")}
    if res.labeled:
        cfg["precision"] = st.column_config.ProgressColumn("precision", min_value=0, max_value=1, format="%.2f")
    st.dataframe(rs, hide_index=True, use_container_width=True, column_config=cfg)
    st.markdown("**How risk is built:** each rule hit has a weight; hits combine as `1 − Π(1 − w)`, so two independent weak signals "
                "can outrank one strong one. The anomaly model then adds up to its influence setting. Change the weights in the sidebar and watch the queue move.")
    if res.labeled:
        miss = d[(d["is_fraud"] == 1) & (~d["is_alert"])]
        fp = d[(d["is_fraud"] == 0) & (d["is_alert"])]
        x, y = st.columns(2)
        x.markdown(f"**Missed fraud ({len(miss)})**")
        x.dataframe(miss[["txn_id", "timestamp", "customer_id", "amount", "risk", "fraud_type"]].head(30), hide_index=True)
        y.markdown(f"**False alarms ({len(fp)})**")
        fp = fp.assign(why=fp["reasons"].map(reasons_text))
        y.dataframe(fp[["txn_id", "customer_id", "amount", "risk", "why"]].head(30), hide_index=True)

# ------------------------------------------------------------------ tuner
with tabs[3]:
    st.markdown("**Alert budget:** how many alerts per day an analyst team must handle at each threshold.")
    sw = res.sweep.set_index("threshold")
    st.line_chart(sw[["alerts_per_day"]])
    if res.labeled:
        st.markdown("**Catch rate vs. noise** (precision = how many alerts are real; recall = how much fraud is caught)")
        st.line_chart(sw[["precision", "recall"]])
    st.dataframe(res.sweep, hide_index=True, use_container_width=True)
    st.caption(f"You are currently at threshold {res.threshold:g}. Pick the point your team can staff.")

# ------------------------------------------------------------------ linked
with tabs[4]:
    st.markdown("**Devices shared by several customers** - a classic mule / account-takeover signature.")
    if res.devices.empty:
        st.info("No device is shared by 3+ customers (or the data has no `device_id`).")
    else:
        st.dataframe(res.devices, hide_index=True, use_container_width=True)
        dev = st.selectbox("Inspect device", res.devices["device_id"])
        st.dataframe(d[d["device_id"] == dev][["timestamp", "customer_id", "amount", "merchant", "channel", "risk", "is_alert"]],
                     hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ export
with tabs[5]:
    out = res.alerts.copy()
    out["reasons"] = out["reasons"].map(reasons_text)
    cols = ["txn_id", "timestamp", "customer_id", "amount", "merchant", "category", "channel", "city", "device_id",
            "rule_score", "ml_score", "risk", "tier", "case_id", "reasons"]
    e1, e2 = st.columns(2)
    e1.download_button("⬇ Suspicious transactions (CSV)", out[cols].to_csv(index=False), "suspicious_transactions.csv", "text/csv")
    e2.download_button("⬇ Cases (CSV)", res.cases.to_csv(index=False), "cases.csv", "text/csv")
    if source_name == "synthetic":
        st.download_button("⬇ The synthetic dataset (CSV) - try it in your own tools", raw.to_csv(index=False), "transactions.csv", "text/csv")
    st.markdown("Save this run (and any analyst decisions) to SQLite and build the **Power BI pack**.")
    if st.button("💾 Save run to SQLite + build Power BI pack"):
        conn = connect(HERE / "fraudlens.db")
        dec = st.session_state.get("decisions", {})
        actions = [(r.case_id, *dec[f"{r.customer_id}|{r.start}"]) for r in res.cases.itertuples()
                   if f"{r.customer_id}|{r.start}" in dec]
        run_id = save_run(conn, res, source_name, actions)
        with tempfile.TemporaryDirectory() as tmp:
            files = export_powerbi(conn, tmp, run_id)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in files:
                    z.write(f, f.name)
                dax = HERE / "powerbi" / "measures.dax"
                if dax.exists():
                    z.write(dax, "measures.dax")
        conn.close()
        st.session_state["pbi_zip"] = buf.getvalue()
        st.success(f"Saved as run #{run_id} in fraudlens.db ({len(actions)} analyst decision(s)).")
    if "pbi_zip" in st.session_state:
        st.download_button("⬇ Power BI pack (.zip)", st.session_state["pbi_zip"], "fraudlens_powerbi_pack.zip", "application/zip")
