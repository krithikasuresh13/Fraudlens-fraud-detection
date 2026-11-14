"""Command-line run.

    python run_demo.py                          # generate synthetic data and screen it
    python run_demo.py --csv my_transactions.csv --threshold 45
"""
import argparse
from pathlib import Path

import pandas as pd

from fraudlens.datagen import generate
from fraudlens.db import connect, export_powerbi, save_run
from fraudlens.engine import reasons_text, run_pipeline

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="your transactions (needs timestamp, customer_id, amount)")
    ap.add_argument("--threshold", type=float, default=40)
    ap.add_argument("--db", default=HERE / "fraudlens.db")
    ap.add_argument("--out", default=HERE / "powerbi/data")
    a = ap.parse_args()

    if a.csv:
        raw, source = pd.read_csv(a.csv), a.csv
    else:
        raw, source = generate(), "synthetic"
        (HERE / "data").mkdir(exist_ok=True)
        raw.to_csv(HERE / "data/transactions.csv", index=False)

    res = run_pipeline(raw, threshold=a.threshold)
    m = res.metrics
    print(f"\n{m['transactions']:,} transactions, {m['customers']} customers, {m['days']} days")
    print(f"{m['alerts']} alerts ({m['alert_rate']}% of volume) in {m['cases']} cases; "
          f"{m['amount_flagged']:,.0f} flagged; {m['high']} high-risk")
    if res.labeled:
        print(f"vs. labels: precision {m['precision']:.1%}, recall {m['recall']:.1%}, F1 {m['f1']:.2f}, "
              f"case precision {m.get('case_precision', 0):.1%}")
    print("\nTop 8 cases to work first:")
    print(res.cases.head(8)[["case_id", "customer_id", "n_alerts", "total_amount", "case_risk", "rules"]].to_string(index=False))
    print("\nWhy the top alert fired:")
    top = res.alerts.iloc[0]
    print(f"  {top['txn_id']} risk {top['risk']}: {reasons_text(top['reasons'])}")
    print("\nRule scorecard:")
    print(res.rule_summary.to_string(index=False))

    conn = connect(a.db)
    run_id = save_run(conn, res, source)
    files = export_powerbi(conn, a.out, run_id)
    print(f"\nSaved run #{run_id} to {Path(a.db).name}; Power BI CSVs: {len(files)} files in {a.out}")


if __name__ == "__main__":
    main()
