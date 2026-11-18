"""Rejeu SQLite : compare features/probas/décisions de production avec celles de l'entraînement,
et mesure la latence. Lancer : python test_replay.py [--cards 300]"""
import argparse
import sqlite3
import time
from datetime import datetime

import joblib
import numpy as np

from features import BEHAVIOR_FEATURES, TZ
from inference import BehaviorScorer, ensure_schema, fetch_history
from train import DATA, MODEL, load_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=int, default=300)
    a = ap.parse_args()

    df = load_data(DATA).sort_values("transaction_id")
    rng = np.random.default_rng(0)
    fraud_cards = df.loc[df.fraud == 1, "card_id"].unique()
    other_cards = np.setdiff1d(df.card_id.unique(), fraud_cards)
    cards = np.concatenate([rng.choice(fraud_cards, min(a.cards // 2, len(fraud_cards)), replace=False),
                            rng.choice(other_cards, a.cards // 2, replace=False)])
    sub = df[df.card_id.isin(cards)]

    scorer, model = BehaviorScorer(), joblib.load(MODEL)
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)

    feat_diff, probs, lat = [], [], []
    for r in sub.itertuples(index=False):
        ts = int(datetime.fromisoformat(r.timestamp).replace(tzinfo=TZ).timestamp())
        t_start = time.perf_counter_ns()
        hist = fetch_history(conn, r.card_id, ts)
        p, block, f = scorer.score(float(r.amount), ts, hist)
        lat.append((time.perf_counter_ns() - t_start) / 1e6)
        conn.execute("INSERT INTO transactions(card_id, ts, amount) VALUES (?, ?, ?)", (r.card_id, ts, r.amount))
        feat_diff.append(max(abs(f[k] - getattr(r, k)) for k in BEHAVIOR_FEATURES))
        probs.append(p)

    sk = model.predict_proba(sub[BEHAVIOR_FEATURES].astype(np.float32))[:, 1]
    probs = np.array(probs)
    far = np.abs(sk - scorer.threshold) > 1e-5
    dec_ok = ((probs >= scorer.threshold) == (sk >= scorer.threshold))[far].all()
    lat = np.array(lat)
    print(f"{len(sub)} transactions rejouées ({len(cards)} cartes, dont {len(cards) // 2} avec fraude)")
    print(f"écart max features  prod vs CSV : {max(feat_diff):.2e}")
    print(f"écart max probas ONNX vs sklearn : {np.abs(probs - sk).max():.2e}")
    print(f"décisions identiques : {dec_ok}")
    print(f"latence SQLite+features+ONNX : p50={np.percentile(lat, 50):.2f} ms | "
          f"p99={np.percentile(lat, 99):.2f} ms | max={lat.max():.2f} ms")
    assert max(feat_diff) < 1e-6, "DÉCALAGE entraînement/production sur les features"
    assert np.abs(probs - sk).max() < 1e-5 and dec_ok, "DÉCALAGE sur les scores ou les décisions"
    print("OK : aucun décalage entraînement/production sur l'échantillon.")