"""Génère dataset_ATM.csv, cohérent avec le simulateur ATM (schéma du cahier des charges, §10).
Kaggle (V1..V28 anonymisées) n'a pas de card_id/ATM/localisation : on simule donc des cartes
avec profil d'usage, puis on injecte 3 types de fraude :
  - behavior          : montant anormal et/ou rafale de retraits (visible par Fog A)
  - impossible_travel : ville lointaine en quelques minutes, montant normal (visible par Fog B)
  - distributed       : les deux à la fois (corrélation A+B+C)

V2 (hour_of_day) : les heures sont générées dans le fuseau TZ de features.py.
  - 10 % des cartes légitimes ont un profil "noctambule" (--night-cards) : l'heure seule
    ne doit pas suffire à désigner une fraude ;
  - une part des fraudes "behavior" a lieu la nuit suivant le vol (--night-fraud).
    HYPOTHÈSE du simulateur ; mettre --night-fraud 0 pour mesurer son effet.
"""
import argparse
import random
from datetime import datetime, timedelta

import pandas as pd

from features import CITIES, TZ, behavior_features, context_features, haversine_km

CITY_LIST = list(CITIES)
START = datetime(2026, 8, 1, tzinfo=TZ)


def atm_for(city, rng):
    return f"ATM_{city.upper()}_{rng.randint(1, 3):02d}"


def far_city(city, rng, min_km=150):
    far = [c for c in CITY_LIST if haversine_km(CITIES[c], CITIES[city]) >= min_km]
    return rng.choice(far) if far else max(CITY_LIST, key=lambda c: haversine_km(CITIES[c], CITIES[city]))


HOUR_W = [0.2] * 7 + [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0.5, 0.3, 0.2]   # poids par heure (0..23)
NIGHT_HOUR_W = [1.5] * 5 + [0.3] * 2 + [0.5] * 10 + [2] * 7                       # profil noctambule


def legit_timeline(rng, days, night_cards=0.10):
    """Transactions légitimes d'une carte, jour par jour : [(ts, city, amount)]."""
    home = rng.choice(CITY_LIST)
    mu = rng.uniform(40, 150)
    hour_w = NIGHT_HOUR_W if rng.random() < night_cards else HOUR_W
    out, city, last_t = [], home, None
    for d in range(days):
        n = rng.choices([0, 1, 2, 3], [0.25, 0.4, 0.25, 0.1])[0]
        times = sorted(START.timestamp() + d * 86400 + rng.choices(range(24), hour_w)[0] * 3600
                       + rng.randint(0, 3599) for _ in range(n))
        for t in times:
            amount = min(max(round(rng.lognormvariate(0, 0.35) * mu / 10) * 10, 10), 600)
            if rng.random() < 0.03:                       # bruit : gros retrait légitime
                amount = min(amount * rng.uniform(3, 6), 900)
            new = rng.choice(CITY_LIST) if rng.random() < 0.08 else home   # déplacement ponctuel
            if last_t is not None and new != city:        # trajet physiquement plausible (<= ~80 km/h)
                travel = haversine_km(CITIES[city], CITIES[new]) / 80 * 3600
                t = max(t, last_t + travel * rng.uniform(1.2, 2.0))
            city, last_t = new, max(t, last_t or 0.0)
            out.append((t, city, round(amount)))
            if rng.random() < 0.04:                       # bruit : 2e retrait rapproché
                t2 = t + rng.uniform(60, 200)
                out.append((t2, city, round(min(amount * rng.uniform(0.3, 1.0), 600))))
                last_t = t2
            if rng.random() < 0.10:                       # bruit : 2e passage légitime dans l'heure
                t3 = last_t + rng.uniform(300, 5400)
                a3 = min(max(round(rng.lognormvariate(0, 0.35) * mu / 10) * 10, 10), 600)
                out.append((t3, city, a3))
                last_t = t3
    return out, home, mu


def next_night(ts0, rng):
    """Instant aléatoire entre 0h et 5h la nuit suivant ts0 (fuseau TZ)."""
    d = datetime.fromtimestamp(ts0, TZ).replace(hour=0, minute=0, second=0) + timedelta(days=1)
    return d.timestamp() + rng.uniform(0, 5 * 3600)


def inject_fraud(rng, legit, mu, night_fraud=0.35):
    """Retourne (timeline légitime filtrée, liste de fraudes (ts, city, amount, type))."""
    frauds, blocked = [], []
    for _ in range(rng.choices([0, 1, 2], [0.45, 0.4, 0.15])[0]):
        i = rng.randint(10, len(legit) - 2)
        ts0, city0, _ = legit[i]
        kind = rng.choices(["behavior", "impossible_travel", "distributed"], [0.4, 0.35, 0.25])[0]
        if kind == "behavior":
            t = next_night(ts0, rng) if rng.random() < night_fraud else ts0 + rng.uniform(120, 6 * 3600)
            city, k = city0, rng.randint(1, 6)
        else:
            city = far_city(city0, rng)
            max_gap = haversine_km(CITIES[city0], CITIES[city]) / 130 * 3600     # vitesse implicite >= 130 km/h
            t = ts0 + rng.uniform(180, max(max_gap - 60, 240))
            k = 1 if kind == "impossible_travel" else rng.randint(2, 6)
        for _ in range(k):
            if kind == "impossible_travel":               # montant NORMAL : invisible pour Fog A
                amount = mu * rng.uniform(0.7, 2.0)
            else:
                amount = mu * rng.uniform(2.5, 12) if rng.random() < 0.85 else mu * rng.uniform(1, 2.5)
            frauds.append((t, city, round(min(max(amount, 20), 1000)), kind))
            t += rng.uniform(15, 80)
        blocked.append((ts0, t + 12 * 3600))              # carte bloquée ~12h après la fraude
    keep = [x for x in legit if not any(a < x[0] < b for a, b in blocked)]
    return keep, frauds


def build(n_cards, days, seed, night_cards=0.10, night_fraud=0.35):
    rng = random.Random(seed)
    rows = []
    for c in range(n_cards):
        legit, home, mu = legit_timeline(rng, days, night_cards)
        legit, frauds = inject_fraud(rng, legit, mu, night_fraud)
        events = [(t, city, a, "legit") for t, city, a in legit] + frauds
        events.sort(key=lambda e: e[0])
        hist, prev_loc, prev_ts = [], None, None
        for t, city, amount, kind in events:
            f = behavior_features(amount, t, hist)
            f.update(context_features(city, t, prev_loc, prev_ts))
            rows.append({
                "transaction_id": f"TX{len(rows) + 1:07d}", "card_id": f"CARD{c + 1:05d}",
                "amount": amount,
                "timestamp": datetime.fromtimestamp(t, TZ).replace(tzinfo=None).isoformat(timespec="seconds"),
                "atm_id": atm_for(city, rng), "location": city,
                **{k: v for k, v in f.items() if k != "amount"},
                "fraud": int(kind != "legit"), "fraud_type": kind,
            })
            hist.append((t, amount))
            prev_loc, prev_ts = city, t
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cards", type=int, default=1500)
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--night-cards", type=float, default=0.10, help="part de cartes légitimes noctambules")
    p.add_argument("--night-fraud", type=float, default=0.35, help="part des fraudes behavior nocturnes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/dataset_ATM.csv")
    a = p.parse_args()
    df = build(a.cards, a.days, a.seed, a.night_cards, a.night_fraud)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} transactions, {df.card_id.nunique()} cartes, fraude = {df.fraud.mean():.2%}")
    print(df.fraud_type.value_counts().to_string())
