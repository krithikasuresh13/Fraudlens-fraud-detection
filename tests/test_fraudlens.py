"""Run with:  python -m unittest discover tests"""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens.datagen import generate  # noqa: E402
from fraudlens.db import connect, export_powerbi, save_run  # noqa: E402
from fraudlens.engine import apply_rules, compute_features, prepare, run_pipeline  # noqa: E402

T0 = pd.Timestamp("2026-08-01 10:00")


def history(cid="C1", n=40, amount=500.0, device="D1", city=("Bengaluru", 12.97, 77.59)):
    """A calm customer: one purchase every ~day around the same amount."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "timestamp": [T0 - pd.Timedelta(days=n - i, hours=int(rng.integers(0, 5))) for i in range(n)],
        "customer_id": cid, "amount": amount * rng.uniform(0.8, 1.25, n), "device_id": device, "channel": "POS",
        "city": city[0], "lat": city[1], "lon": city[2], "country": "IN", "merchant": "Shop", "category": "Grocery"})


def hits(df):
    d = apply_rules(compute_features(prepare(df)))
    return {r: bool(d[f"r_{r}"].any()) for r in
            ["AMOUNT_SPIKE", "VELOCITY_BURST", "IMPOSSIBLE_TRAVEL", "CARD_TEST_BURST", "CARD_TEST_CASHOUT",
             "NEW_DEVICE_HIGH", "ODD_HOUR_HIGH", "SHARED_DEVICE", "FIRST_FOREIGN"]}


def add(df, **rows):
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


class Rules(unittest.TestCase):
    def test_calm_customer_triggers_nothing(self):
        self.assertFalse(any(hits(history()).values()))

    def test_amount_spike(self):
        df = add(history(), timestamp=[T0], customer_id="C1", amount=[40000], device_id="D1", channel="POS",
                 city="Bengaluru", lat=12.97, lon=77.59, country="IN")
        self.assertTrue(hits(df)["AMOUNT_SPIKE"])

    def test_velocity_burst(self):
        n = 6
        df = add(history(), timestamp=[T0 + pd.Timedelta(minutes=i) for i in range(n)], customer_id="C1", amount=500.0,
                 device_id="D1", channel="ONLINE", city="Bengaluru", lat=12.97, lon=77.59, country="IN")
        self.assertTrue(hits(df)["VELOCITY_BURST"])

    def test_impossible_travel(self):
        df = add(history(), timestamp=[T0, T0 + pd.Timedelta(minutes=30)], customer_id="C1", amount=500.0, device_id="D1",
                 channel="POS", city=["Bengaluru", "Delhi"], lat=[12.97, 28.61], lon=[77.59, 77.21], country="IN")
        self.assertTrue(hits(df)["IMPOSSIBLE_TRAVEL"])

    def test_normal_travel_is_fine(self):
        df = add(history(), timestamp=[T0, T0 + pd.Timedelta(hours=30)], customer_id="C1", amount=500.0, device_id="D1",
                 channel="POS", city=["Bengaluru", "Delhi"], lat=[12.97, 28.61], lon=[77.59, 77.21], country="IN")
        self.assertFalse(hits(df)["IMPOSSIBLE_TRAVEL"])

    def test_card_testing_then_cashout(self):
        df = add(history(), timestamp=[T0 + pd.Timedelta(minutes=i) for i in range(4)] + [T0 + pd.Timedelta(minutes=12)],
                 customer_id="C1", amount=[5, 8, 3, 10, 4500], device_id="D1", channel="ONLINE", city="Bengaluru",
                 lat=12.97, lon=77.59, country="IN")
        h = hits(df)
        self.assertTrue(h["CARD_TEST_BURST"])
        self.assertTrue(h["CARD_TEST_CASHOUT"])

    def test_shared_device(self):
        df = pd.concat([history(cid=f"C{i}", n=10, device="D-SHARED") for i in range(5)], ignore_index=True)
        self.assertTrue(hits(df)["SHARED_DEVICE"])

    def test_first_foreign(self):
        df = add(history(), timestamp=[T0], customer_id="C1", amount=900, device_id="D1", channel="ONLINE",
                 city="Dubai", lat=25.2, lon=55.27, country="AE")
        self.assertTrue(hits(df)["FIRST_FOREIGN"])

    def test_works_without_optional_columns(self):
        df = history()[["timestamp", "customer_id", "amount"]]
        res = run_pipeline(df)                       # no lat/lon/device/channel: must not crash
        self.assertEqual(len(res.df), len(df))

    def test_missing_required_column(self):
        with self.assertRaises(ValueError):
            run_pipeline(pd.DataFrame({"customer_id": ["a"], "amount": [1]}))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate(n_customers=150, days=60, seed=3)
        cls.res = run_pipeline(cls.raw)

    def test_catches_most_planted_fraud(self):
        self.assertGreater(self.res.metrics["recall"], 0.8)
        self.assertGreater(self.res.metrics["precision"], 0.6)

    def test_detector_ignores_labels(self):
        # shuffling the hidden labels must not change any score
        shuffled = self.raw.assign(is_fraud=np.random.default_rng(0).permutation(self.raw["is_fraud"].values))
        other = run_pipeline(shuffled)
        a = self.res.df.set_index("txn_id")["risk"]
        b = other.df.set_index("txn_id")["risk"].reindex(a.index)
        pd.testing.assert_series_equal(a, b)

    def test_cases_partition_alerts(self):
        self.assertEqual(int(self.res.cases["n_alerts"].sum()), self.res.metrics["alerts"])

    def test_every_alert_has_a_reason(self):
        self.assertTrue(all(len(r) > 0 for r in self.res.alerts["reasons"]))

    def test_sql_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "t.db")
            run_id = save_run(conn, self.res, "test", [(self.res.cases["case_id"].iat[0], "confirm_fraud", "ok")])
            n = conn.execute("SELECT COUNT(*) FROM vw_alerts WHERE run_id=?", (run_id,)).fetchone()[0]
            self.assertEqual(n, self.res.metrics["alerts"])
            status = conn.execute("SELECT status FROM vw_case_queue WHERE run_id=? ORDER BY priority DESC LIMIT 1", (run_id,)).fetchone()[0]
            self.assertEqual(status, "confirm_fraud")
            files = export_powerbi(conn, Path(tmp) / "out", run_id)
            self.assertEqual(len(files), 8)
            conn.close()


if __name__ == "__main__":
    unittest.main()
