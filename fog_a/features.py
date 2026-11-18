"""Feature engineering PARTAGÉ : utilisé par le dataset builder ET par Fog A / Fog B.
Une seule implémentation =>  pas de décalage entre entraînement et production.

BEHAVIOR_FEATURES est la SOURCE UNIQUE de la liste (et de l'ORDRE) des features de Fog A.
Toute modification impose : generate_dataset.py -> train.py -> evaluate.py -> export_onnx.py,
et Fog A doit envoyer le vecteur dans ce même ordre (voir FEATURES_VERSION dans le .onnx).

Critères d'admission d'une feature dans Fog A :
  1. calculable depuis la transaction courante + SQLite A ;
  2. n'utilise que le PASSÉ de la carte (pas de fuite) ;
  3. décrit le comportement de la carte (sinon empiète sur Fog B / Fog C) ;
  4. générée de façon réaliste par le simulateur.
"""
import math
from datetime import datetime, timedelta, timezone

FEATURES_VERSION = "v2"
BEHAVIOR_FEATURES = [
    # v1
    "amount",
    "transactions_last_5min",
    "average_amount",
    "amount_deviation",
    "time_since_previous",
    # v2
    "transactions_last_30min",
    "transactions_last_24h",
    "max_amount_7d",
    "hour_of_day",
]
HISTORY_WINDOW = 20          # nb de transactions passées pour average_amount
BURST_WINDOW_S = 300         # 5 minutes
WINDOW_30MIN_S = 30 * 60
WINDOW_24H_S = 24 * 3600
WINDOW_7D_S = 7 * 24 * 3600
MAX_GAP_S = 7 * 24 * 3600    # plafond / valeur par défaut (première transaction)
TZ = timezone(timedelta(hours=1))   # heure locale Tunisie (UTC+1, pas d'heure d'été)

# Coordonnées (lat, lon) des villes
CITIES = {
    "Tunis": (36.8065, 10.1815), "Sfax": (34.7406, 10.7603),
    "Sousse": (35.8256, 10.6369), "Monastir": (35.7643, 10.8113),
    "Nabeul": (36.4513, 10.7357), "Bizerte": (37.2744, 9.8739),
    "Gabes": (33.8815, 10.0982), "Kairouan": (35.6781, 10.0963),
    "Gafsa": (34.4250, 8.7842), "Mahdia": (35.5047, 11.0622),
}


def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def hour_of_day(ts):
    """Heure locale (0..23) d'un epoch, en fuseau fixe TZ (indépendant de la machine)."""
    return datetime.fromtimestamp(ts, TZ).hour


def _amounts_in_window(history, ts, seconds):
    """Montants des transactions PASSÉES avec 0 <= ts - t <= seconds (history triée)."""
    out = []
    for t, a in reversed(history):
        dt = ts - t
        if dt > seconds:
            break
        if dt >= 0:
            out.append(a)
    return out


def behavior_features(amount, ts, history):
    """amount: float | ts: epoch secondes | history: [(ts, amount), ...] triée, transactions
    PRÉCÉDENTES de la même carte. Retourne les features de Fog A (clés = BEHAVIOR_FEATURES).
    Équivalents SQLite A : COUNT(*) / MAX(amount) WHERE card_id=? AND ts BETWEEN ?-fenêtre AND ?"""
    recent = history[-HISTORY_WINDOW:]
    if recent:
        avg = sum(a for _, a in recent) / len(recent)
        gap = min(max(ts - recent[-1][0], 0.0), MAX_GAP_S)
    else:                                   # cold start
        avg, gap = amount, MAX_GAP_S
    last_7d = _amounts_in_window(history, ts, WINDOW_7D_S)
    return {
        "amount": float(amount),
        "transactions_last_5min": len(_amounts_in_window(history, ts, BURST_WINDOW_S)),
        "average_amount": avg,
        "amount_deviation": (amount - avg) / max(avg, 1.0),   # écart relatif
        "time_since_previous": gap,
        "transactions_last_30min": len(_amounts_in_window(history, ts, WINDOW_30MIN_S)),
        "transactions_last_24h": len(_amounts_in_window(history, ts, WINDOW_24H_S)),
        "max_amount_7d": float(max(last_7d)) if last_7d else 0.0,   # 0 = aucun retrait sur 7 j
        "hour_of_day": hour_of_day(ts),
    }


def context_features(loc, ts, prev_loc, prev_ts):
    """Features de Fog B (règles) : distance (km) et vitesse implicite (km/h)."""
    if prev_loc is None:
        return {"distance_from_previous_atm": 0.0, "speed": 0.0}
    d = haversine_km(CITIES[loc], CITIES[prev_loc])
    hours = max(ts - prev_ts, 1.0) / 3600.0
    return {"distance_from_previous_atm": d, "speed": min(d / hours, 5000.0)}