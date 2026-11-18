"""Module d'inférence pour Fog A. `history` = fenêtre 7 jours ∪ 20 dernières transactions, triée
(fetch_history le fait). Enregistrer TOUTES les transactions, refusées comprises. En FastAPI, appeler
`score` depuis un endpoint `def` (sess.run est bloquant)."""
import json
import math
import sqlite3

import numpy as np
import onnxruntime as ort

from features import BEHAVIOR_FEATURES, FEATURES_VERSION, behavior_features

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id TEXT NOT NULL, ts INTEGER NOT NULL, amount REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_tx_card_ts ON transactions(card_id, ts);
"""

HISTORY_SQL = """
SELECT ts, amount FROM transactions
WHERE card_id = :c AND ts <= :now AND (
      ts >= :now - 604800
   OR id IN (SELECT id FROM transactions WHERE card_id = :c AND ts <= :now
             ORDER BY ts DESC, id DESC LIMIT 20))
ORDER BY ts ASC, id ASC
"""


def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)


def fetch_history(conn: sqlite3.Connection, card_id: str, now_ts: int):
    """À appeler AVANT d'insérer la transaction courante."""
    return conn.execute(HISTORY_SQL, {"c": card_id, "now": int(now_ts)}).fetchall()


class BehaviorScorer:
    def __init__(self, model_path="artifacts/model.onnx"):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        if (json.loads(meta.get("features", "null")) != BEHAVIOR_FEATURES
                or meta.get("features_version") != FEATURES_VERSION):
            raise RuntimeError(f"{model_path} (features {meta.get('features_version')}) incohérent avec "
                               f"features.py ({FEATURES_VERSION}) : réentraîner et réexporter")
        self.threshold = float(meta["threshold"])
        self.metadata = dict(meta)

    def score(self, amount, ts, history):
        """history: [(ts_epoch_secondes, amount), ...]. Retourne (score, block: bool, features)."""
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"amount invalide : {amount}")
        if not (1e9 < ts < 4e9):
            raise ValueError("ts doit être en SECONDES epoch (pas en millisecondes)")
        history = sorted(history)
        f = behavior_features(amount, ts, history)
        x = np.array([[f[k] for k in BEHAVIOR_FEATURES]], dtype=np.float32)
        p = float(self.sess.run(["probabilities"], {self.input_name: x})[0][0, 1])
        return p, p >= self.threshold, f


if __name__ == "__main__":
    s = BehaviorScorer()
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    t0 = 1_790_000_000
    past = [(t0 - 86400 * k, a) for k, a in zip(range(10, 0, -1), [60, 80, 70, 90, 50, 100, 70, 60, 80, 70])]
    past += [(t0 - 170, 90), (t0 - 120, 90), (t0 - 70, 100), (t0 - 30, 90)]
    conn.executemany("INSERT INTO transactions(card_id, ts, amount) VALUES ('CARD001', ?, ?)", past)
    for label, amount, ts in [("normal (J+2)", 70, t0 + 2 * 86400), ("rafale + 900 DT", 900, t0)]:
        p, block, _ = s.score(amount, ts, fetch_history(conn, "CARD001", ts))
        print(f"{label:16s}: score={p:.3f} -> {'BLOCK' if block else 'ALLOW'} (seuil {s.threshold})")