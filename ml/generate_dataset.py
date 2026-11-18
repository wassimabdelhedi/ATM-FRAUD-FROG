"""Simulateur ATM (SYNTHÉTIQUE). 4 types de fraude : behavior, card_testing (petits montants en rafale),
impossible_travel (Fog B), distributed. --fraud-card-rate contrôle la prévalence (0.045 => ~0.3 %),
--difficulty [0..1] fait chevaucher fraude/légitime. Montants multiples de 10, timestamps entiers,
colonne fraud_episode_id."""
import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd

from features import CITIES, TZ, behavior_features, context_features, haversine_km

CITY_LIST = list(CITIES)
START = datetime(2026, 8, 1, tzinfo=TZ)
HOUR_W = [0.2] * 7 + [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0.5, 0.3, 0.2]
NIGHT_HOUR_W = [1.5] * 5 + [0.3] * 2 + [0.5] * 10 + [2] * 7
KINDS = ["behavior", "card_testing", "impossible_travel", "distributed"]
KIND_W = [0.30, 0.30, 0.25, 0.15]          # HYPOTHÈSE : répartition des types de fraude


def atm_round(x, lo=10, hi=None):
    v = max(lo, int(round(x / 10.0)) * 10)
    return min(v, hi) if hi else v


def atm_for(city, rng):
    return f"ATM_{city.upper()}_{rng.randint(1, 3):02d}"


def far_city(city, rng, min_km=150):
    far = [c for c in CITY_LIST if haversine_km(CITIES[c], CITIES[city]) >= min_km]
    return rng.choice(far) if far else max(CITY_LIST, key=lambda c: haversine_km(CITIES[c], CITIES[city]))


def legit_timeline(rng, days, night_cards=0.10):
    home = rng.choice(CITY_LIST)
    mu = rng.uniform(40, 150)
    hour_w = NIGHT_HOUR_W if rng.random() < night_cards else HOUR_W
    out, city, last_t = [], home, None
    for d in range(days):
        n = rng.choices([0, 1, 2, 3], [0.25, 0.4, 0.25, 0.1])[0]
        times = sorted(START.timestamp() + d * 86400 + rng.choices(range(24), hour_w)[0] * 3600
                       + rng.randint(0, 3599) for _ in range(n))
        for t in times:
            amount = atm_round(rng.lognormvariate(0, 0.35) * mu, hi=600)
            if rng.random() < 0.08:
                amount = rng.choice([10, 20, 30])
            if rng.random() < 0.03:
                amount = atm_round(amount * rng.uniform(3, 6), hi=900)
            new = rng.choice(CITY_LIST) if rng.random() < 0.08 else home
            if last_t is not None and new != city:
                travel = haversine_km(CITIES[city], CITIES[new]) / 80 * 3600
                t = max(t, last_t + travel * rng.uniform(1.2, 2.0))
            city, last_t = new, max(t, last_t or 0.0)
            out.append((t, city, amount))
            if rng.random() < 0.04:
                t2 = t + rng.uniform(60, 200)
                out.append((t2, city, atm_round(amount * rng.uniform(0.3, 1.0), hi=600)))
                last_t = t2
            if rng.random() < 0.10:
                t3 = last_t + rng.uniform(300, 5400)
                out.append((t3, city, atm_round(rng.lognormvariate(0, 0.35) * mu, hi=600)))
                last_t = t3
    return out, home, mu


def next_night(ts0, rng):
    d = datetime.fromtimestamp(ts0, TZ).replace(hour=0, minute=0, second=0) + timedelta(days=1)
    return d.timestamp() + rng.uniform(0, 5 * 3600)


def inject_fraud(rng, legit, mu, night_fraud=0.35, difficulty=0.5, card_rate=0.045):
    frauds, blocked = [], []
    if len(legit) < 6:
        return legit, frauds
    d = difficulty
    n_ep = rng.choices([0, 1, 2], [1 - card_rate, card_rate * 0.85, card_rate * 0.15])[0]
    for ep in range(n_ep):
        cand = [i for i in range(2, len(legit) - 1)
                if not any(a < legit[i][0] < b for a, b in blocked)]
        if not cand:
            break
        ts0, city0, _ = legit[rng.choice(cand)]
        kind = rng.choices(KINDS, KIND_W)[0]
        gap_hi = 80 + 220 * d
        if kind in ("behavior", "card_testing"):
            t = next_night(ts0, rng) if rng.random() < night_fraud else ts0 + rng.uniform(120, 6 * 3600)
            city = city0
            k = rng.randint(1, 6) if kind == "behavior" else rng.randint(3, 8)
        else:
            city = far_city(city0, rng)
            min_speed = 130 - 50 * d
            max_gap = haversine_km(CITIES[city0], CITIES[city]) / min_speed * 3600
            t = ts0 + rng.uniform(180, max(max_gap - 60, 240))
            k = 1 if kind == "impossible_travel" else rng.randint(2, 6)
        amounts = []
        for _ in range(k):
            if kind == "impossible_travel":
                a = mu * rng.uniform(0.7, 2.0)
            elif kind == "card_testing":
                a = rng.choice([10, 10, 20, 30])
            else:
                a = (mu * rng.uniform(2.5 - 1.5 * d, 12 - 6 * d) if rng.random() < 0.85
                     else mu * rng.uniform(1, 2.5))
            amounts.append(atm_round(a, hi=1000))
        if kind == "card_testing" and rng.random() < 0.5:
            amounts.append(atm_round(mu * rng.uniform(2, 6), hi=1000))
        for a in amounts:
            frauds.append((t, city, a, kind, ep))
            t += rng.uniform(15, gap_hi)
        blocked.append((ts0, t + 12 * 3600))
    keep = [x for x in legit if not any(a < x[0] < b for a, b in blocked)]
    return keep, frauds


def build(n_cards, days, seed, night_cards=0.10, night_fraud=0.35, difficulty=0.5, card_rate=0.045):
    rng = random.Random(seed)
    rows = []
    for c in range(n_cards):
        legit, home, mu = legit_timeline(rng, days, night_cards)
        legit, frauds = inject_fraud(rng, legit, mu, night_fraud, difficulty, card_rate)
        events = [(t, city, a, "legit", "") for t, city, a in legit]
        events += [(t, city, a, kind, f"EP{c + 1:05d}_{ep}") for t, city, a, kind, ep in frauds]
        events.sort(key=lambda e: e[0])
        hist, prev_loc, prev_ts = [], None, None
        for t, city, amount, kind, ep in events:
            t = int(t)
            f = behavior_features(amount, t, hist)
            f.update(context_features(city, t, prev_loc, prev_ts))
            rows.append({
                "transaction_id": f"TX{len(rows) + 1:07d}", "card_id": f"CARD{c + 1:05d}",
                "amount": amount,
                "timestamp": datetime.fromtimestamp(t, TZ).replace(tzinfo=None).isoformat(timespec="seconds"),
                "atm_id": atm_for(city, rng), "location": city,
                **{k: v for k, v in f.items() if k != "amount"},
                "fraud": int(kind != "legit"), "fraud_type": kind, "fraud_episode_id": ep,
            })
            hist.append((t, amount))
            prev_loc, prev_ts = city, t
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cards", type=int, default=6000)
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--fraud-card-rate", type=float, default=0.045)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--night-cards", type=float, default=0.10)
    p.add_argument("--night-fraud", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/dataset_ATM.csv")
    a = p.parse_args()
    df = build(a.cards, a.days, a.seed, a.night_cards, a.night_fraud, a.difficulty, a.fraud_card_rate)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    n_ep = df.loc[df.fraud == 1, "fraud_episode_id"].nunique()
    print(f"{len(df)} transactions, {df.card_id.nunique()} cartes, "
          f"fraude = {df.fraud.mean():.3%} ({int(df.fraud.sum())} tx, {n_ep} épisodes)")
    print(df.fraud_type.value_counts().to_string())