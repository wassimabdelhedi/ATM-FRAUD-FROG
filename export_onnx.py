"""Export du Random Forest en ONNX + vérification de parité sklearn <-> ONNX Runtime."""
import json

import joblib
import numpy as np
import onnxruntime as ort
# pyrefly: ignore [missing-import]
from skl2onnx import to_onnx

from features import BEHAVIOR_FEATURES, FEATURES_VERSION
from train import MODEL, load_split

ONNX = "artifacts/model.onnx"

if __name__ == "__main__":
    model = joblib.load(MODEL)
    _, test = load_split()
    X = test[BEHAVIOR_FEATURES].to_numpy(dtype=np.float32)
    onx = to_onnx(model, X[:1], options={id(model): {"zipmap": False}}, target_opset=17)
    for k, v in {"features_version": FEATURES_VERSION, "features": json.dumps(BEHAVIOR_FEATURES)}.items():
        p = onx.metadata_props.add()
        p.key, p.value = k, v
    open(ONNX, "wb").write(onx.SerializeToString())

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    onnx_p = sess.run(None, {sess.get_inputs()[0].name: X})[1][:, 1]
    skl_p = model.predict_proba(test[BEHAVIOR_FEATURES].astype(np.float32))[:, 1]
    diff = float(np.abs(onnx_p - skl_p).max())
    print(f"ONNX exporté -> {ONNX} | écart max sklearn/ONNX = {diff:.2e}")
    assert diff < 1e-4, "parité sklearn/ONNX non respectée"
