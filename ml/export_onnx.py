"""Export ONNX + parité sklearn/ONNX. Seuil, versions et empreinte du dataset embarqués dans le .onnx."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import skl2onnx
import sklearn
from skl2onnx import to_onnx

from features import BEHAVIOR_FEATURES, FEATURES_VERSION
from train import DATA, MODEL, THRESHOLD_FILE, load_split3

ONNX = Path("artifacts/model.onnx")
PARITY_EPS = 1e-5
EDGE = np.array([                                      # ordre = BEHAVIOR_FEATURES
    [10, 0, 10, 0.0, 604800, 0, 0, 0.0, 3],            # cold start
    [1000, 10, 90, 10.0, 5, 15, 40, 900.0, 3],         # rafale extrême
    [10, 6, 20, -0.5, 15, 8, 12, 30.0, 23],            # card testing
    [20, 0, 20, 0.0, 604800, 0, 0, 0.0, 12],
    [600, 0, 90, 5.0, 86400, 0, 1, 100.0, 12],         # gros retrait isolé
], dtype=np.float32)

if __name__ == "__main__":
    model = joblib.load(MODEL)
    assert list(model.classes_) == [0, 1], model.classes_
    thr = json.load(open(THRESHOLD_FILE))["threshold"]
    _, _, test = load_split3()
    X = np.vstack([test[BEHAVIOR_FEATURES].to_numpy(dtype=np.float32), EDGE])

    onx = to_onnx(model, X[:1], options={id(model): {"zipmap": False}}, target_opset=17)
    meta = {
        "features_version": FEATURES_VERSION,
        "features": json.dumps(BEHAVIOR_FEATURES),
        "threshold": repr(float(thr)),
        "sklearn": sklearn.__version__, "skl2onnx": skl2onnx.__version__,
        "train_data_sha256": hashlib.sha256(Path(DATA).read_bytes()).hexdigest()[:16],
    }
    for k, v in meta.items():
        p = onx.metadata_props.add()
        p.key, p.value = k, v
    ONNX.write_bytes(onx.SerializeToString())

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    onnx_p = sess.run(["probabilities"], {sess.get_inputs()[0].name: X})[0][:, 1]
    skl_p = model.predict_proba(pd.DataFrame(X, columns=BEHAVIOR_FEATURES))[:, 1]
    diff = float(np.abs(onnx_p - skl_p).max())
    print(f"ONNX -> {ONNX} ({ONNX.stat().st_size / 1e6:.1f} Mo) | écart max probabilités = {diff:.2e} | seuil = {thr}")
    assert diff < PARITY_EPS, "parité des probabilités sklearn/ONNX non respectée"
    far = np.abs(skl_p - thr) > PARITY_EPS
    assert ((onnx_p >= thr) == (skl_p >= thr))[far].all(), "parité des DÉCISIONS sklearn/ONNX non respectée"
    print(f"parité des décisions OK sur {int(far.sum())} échantillons ({len(EDGE)} cas limites inclus)")