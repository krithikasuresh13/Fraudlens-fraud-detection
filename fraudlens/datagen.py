"""Synthetic card/UPI transaction generator with labelled fraud patterns.

The detector never sees `is_fraud` / `fraud_type`; they exist only to evaluate it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import haversine

INDIA = {"Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Chennai": (13.08, 80.27),
         "Hyderabad": (17.39, 78.49), "Kolkata": (22.57, 88.36), "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57),
         "Jaipur": (26.91, 75.79), "Kochi": (9.93, 76.27)}
ABROAD = {"Dubai": (25.20, 55.27, "AE"), "Singapore": (1.35, 103.82, "SG"), "London": (51.51, -0.13, "GB")}
CATS = {"Grocery": ["BigBasket", "DMart", "Nature's Basket", "Local Kirana"],
        "Dining": ["Swiggy", "Zomato", "Cafe Coffee Day", "Barbeque Nation"],
        "Fuel": ["Indian Oil", "HP Petrol", "Shell"],
        "Shopping": ["Amazon", "Flipkart", "Myntra", "Reliance Trends"],
        "Travel": ["IRCTC", "Uber", "Ola", "MakeMyTrip"],
        "Bills": ["BESCOM", "Airtel", "Jio", "Tata Power"],
        "Entertainment": ["BookMyShow", "Netflix", "Spotify"]}
CAT_P = np.array([0.28, 0.20, 0.08, 0.17, 0.10, 0.12, 0.05])
HOUR_W = np.array([.2, .1, .1, .1, .2, .5, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 5, 3, 1.5, .6])
FRAUD_MERCHANTS = ["GlobalMart Online", "LuxeGadgets", "GiftCardHub", "QuickElectro", "PrimeDeals24"]


def generate(n_customers: int = 300, days: int = 90, seed: int = 42, start: str = "2026-07-01",
             fraud_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    cities = list(INDIA)
    cids = [f"C{i:04d}" for i in range(1, n_customers + 1)]
    home = rng.choice(cities, n_customers)
    med = np.exp(rng.normal(np.log(900), 0.7, n_customers)).clip(150, 9000)
    rate = rng.uniform(0.25, 1.1, n_customers)
    has2 = rng.random(n_customers) < 0.25
    trav = np.where(rng.random((n_customers, days)) < 0.04, rng.choice(cities, (n_customers, days)), "")
    hp = HOUR_W / HOUR_W.sum()
    cat_names = list(CATS)

    def loc(city):
        lat, lon = INDIA[city] if city in INDIA else ABROAD[city][:2]
        return lat + rng.normal(0, 0.02), lon + rng.normal(0, 0.02), ("IN" if city in INDIA else ABROAD[city][2])

    frames = []
    for i, cid in enumerate(cids):
        n = rng.poisson(rate[i] * days)
        if n == 0:
            continue
        day = rng.integers(0, days, n)
        secs = day * 86400 + rng.choice(24, n, p=hp) * 3600 + rng.integers(0, 3600, n)
        amount = med[i] * rng.lognormal(0, 0.55, n)
        amount = np.where(rng.random(n) < 0.012, amount * rng.uniform(4, 8, n), amount)     # benign large purchases
        cat = rng.choice(cat_names, n, p=CAT_P)
        merch = [rng.choice(CATS[c]) for c in cat]
        city = np.where(trav[i, day] != "", trav[i, day], home[i])
        dev = np.where(has2[i] & (rng.random(n) < 0.3), f"D-{cid}-2", f"D-{cid}-1")
        dev = np.where((day > 30) & (rng.random(n) < 0.004), f"D-{cid}-N", dev)              # benign new phone
        ll = np.array([loc(c) for c in city], dtype=object)
        frames.append(pd.DataFrame({
            "timestamp": t0 + pd.to_timedelta(secs, unit="s"), "customer_id": cid, "amount": amount.round(2).clip(5),
            "merchant": merch, "category": cat, "channel": rng.choice(["POS", "ONLINE", "UPI", "ATM"], n, p=[.45, .30, .20, .05]),
            "city": city, "country": [x[2] for x in ll], "lat": [x[0] for x in ll], "lon": [x[1] for x in ll],
            "device_id": dev, "is_fraud": 0, "fraud_type": ""}))
    base = pd.concat(frames, ignore_index=True)

    idx = {c: k for k, c in enumerate(cids)}
    extra = []

    def row(ts, cid, amount, merchant, category, channel, city, device, ftype, fraud=1):
        lat, lon, ctry = loc(city)
        extra.append(dict(timestamp=ts, customer_id=cid, amount=round(float(amount), 2), merchant=merchant, category=category,
                          channel=channel, city=city, country=ctry, lat=lat, lon=lon, device_id=device,
                          is_fraud=fraud, fraud_type=ftype))

    def pick():
        c = cids[rng.integers(0, n_customers)]
        return c, med[idx[c]], home[idx[c]]

    def when(lo=25, hours=(8, 23)):
        return t0 + pd.Timedelta(days=int(rng.integers(lo, days)), hours=int(rng.integers(*hours)), minutes=int(rng.integers(0, 60)))

    sc = lambda k: max(1, int(round(k * fraud_scale)))
    k = 0
    for _ in range(sc(12)):                                   # 1. amount spikes
        c, m, h = pick(); k += 1
        row(when(), c, m * rng.uniform(15, 40), rng.choice(FRAUD_MERCHANTS), "Electronics", "ONLINE", h,
            f"D-{c}-F{k}" if rng.random() < .5 else f"D-{c}-1", "amount_spike")
    for _ in range(sc(8)):                                    # 2. velocity bursts
        c, m, h = pick(); k += 1; t = when()
        for _j in range(int(rng.integers(6, 10))):
            t += pd.Timedelta(seconds=int(rng.integers(30, 90)))
            row(t, c, m * rng.uniform(2, 4), rng.choice(FRAUD_MERCHANTS), "Gift Cards", "ONLINE", h, f"D-{c}-F{k}", "velocity_burst")
    for _ in range(sc(7)):                                    # 3. impossible travel
        c, m, h = pick(); t = when()
        far = [x for x in cities if haversine(*INDIA[h], *INDIA[x]) > 800]
        row(t, c, m, rng.choice(CATS["Grocery"]), "Grocery", "POS", h, f"D-{c}-1", "", fraud=0)
        row(t + pd.Timedelta(minutes=int(rng.integers(15, 40))), c, m * rng.uniform(4, 7), rng.choice(CATS["Shopping"]),
            "Shopping", "POS", rng.choice(far), f"D-{c}-1", "impossible_travel")
    for _ in range(sc(6)):                                    # 4. card testing -> cash-out
        c, m, h = pick(); k += 1; t = when()
        for _j in range(int(rng.integers(4, 7))):
            t += pd.Timedelta(seconds=int(rng.integers(40, 100)))
            row(t, c, rng.uniform(1, 20), "TestShop Online", "Digital", "ONLINE", h, f"D-{c}-F{k}", "card_testing")
        t += pd.Timedelta(minutes=int(rng.integers(5, 12)))
        row(t, c, rng.uniform(3000, 9000), rng.choice(FRAUD_MERCHANTS), "Electronics", "ONLINE", h, f"D-{c}-F{k}", "card_testing")
    for _ in range(sc(6)):                                    # 5. night-time high value
        c, m, h = pick(); t = when(hours=(2, 5))
        for _j in range(int(rng.integers(1, 3))):
            ch = rng.choice(["ONLINE", "ATM"])
            row(t + pd.Timedelta(minutes=10 * _j), c, m * rng.uniform(6, 12), "ATM Withdrawal" if ch == "ATM" else rng.choice(FRAUD_MERCHANTS),
                "Cash" if ch == "ATM" else "Electronics", ch, h, f"D-{c}-1", "night_high_value")
    ring = [cids[j] for j in rng.choice(n_customers, sc(6) if fraud_scale >= 1 else 4, replace=False)]
    for c in ring:                                            # 6. mule / account-takeover ring on one device
        for _j in range(int(rng.integers(2, 4))):
            row(when(40, (10, 20)), c, med[idx[c]] * rng.uniform(2.5, 5), "QuickPay Wallet", "Wallet Topup", "ONLINE",
                home[idx[c]], "D-RING-01", "device_ring")
    for _ in range(sc(5)):                                    # 7. first foreign transaction
        c, m, h = pick()
        row(when(), c, m * rng.uniform(8, 16), rng.choice(FRAUD_MERCHANTS), "Electronics", "ONLINE", rng.choice(list(ABROAD)),
            f"D-{c}-1", "foreign_first")

    out = pd.concat([base, pd.DataFrame(extra)], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out.insert(0, "txn_id", [f"TXN{i:07d}" for i in range(1, len(out) + 1)])
    return out
