"""Module d'inférence importé par Fog A (FastAPI) : transaction -> behavior_score."""
import json

import numpy as np
import onnxruntime as ort

from features import BEHAVIOR_FEATURES, behavior_features


class BehaviorScorer:
    def __init__(self, model_path="artifacts/model.onnx"):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        if json.loads(meta.get("features", "null")) != BEHAVIOR_FEATURES:
            raise RuntimeError(f"{model_path} entraîné avec {meta.get('features')} != features.py "
                               f"{BEHAVIOR_FEATURES} : réentraîner et réexporter")


    def score(self, amount, ts, history):
        """history: [(ts_epoch, amount), ...] des transactions précédentes de la carte
        (lue dans SQLite A). Retourne (behavior_score, features)."""
        f = behavior_features(amount, ts, history)
        x = np.array([[f[k] for k in BEHAVIOR_FEATURES]], dtype=np.float32)
        return float(self.sess.run(None, {self.input_name: x})[1][0, 1]), f


if __name__ == "__main__":      # démo : cas du cahier des charges (CARD001, 900 DT, rafale)
    s = BehaviorScorer()
    t0 = 1_790_000_000
    hist = [(t0 - 86400 * k, a) for k, a in zip(range(10, 0, -1), [60, 80, 70, 90, 50, 100, 75, 65, 85, 70])]
    hist += [(t0 - 170, 90), (t0 - 120, 95), (t0 - 70, 100), (t0 - 30, 95)]
    print("normal :", s.score(70, t0 + 86400 * 2, hist[:10])[0])
    print("fraude :", s.score(900, t0, hist)[0])
