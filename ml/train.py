"""Entraînement RF Fog A. Split train/val/test PAR CARTE (StratifiedGroupKFold), seuil choisi sur la
VALIDATION par coût, test réservé à evaluate.py, pas de class_weight."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from features import BEHAVIOR_FEATURES

DATA = "data/dataset_ATM.csv"
MODEL = "artifacts/model.joblib"
THRESHOLD_FILE = "artifacts/threshold.json"
SPLIT_SEED = 0
COST_FP = 5.0     # HYPOTHÈSE : coût d'un faux positif en dinars (à justifier dans le rapport)


def load_data(path=DATA):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = [c for c in BEHAVIOR_FEATURES + ["fraud", "card_id", "amount"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} ne contient pas {missing} : relancer generate_dataset.py")
    df = df.dropna(subset=BEHAVIOR_FEATURES + ["fraud"]).reset_index(drop=True)
    df["fraud_episode_id"] = df["fraud_episode_id"].fillna("") if "fraud_episode_id" in df else ""
    return df


def load_split3(path=DATA, seed=SPLIT_SEED, n_folds=5):
    df = load_data(path)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(df, df["fraud"], groups=df["card_id"])]
    test, val = df.iloc[folds[0]], df.iloc[folds[1]]
    train = df.iloc[np.concatenate(folds[2:])]
    return train, val, test


def build_model(seed=42):
    return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                  n_jobs=-1, random_state=seed)


def pick_threshold(y, proba, amounts, c_fp=COST_FP, grid=None):
    y, proba, amounts = np.asarray(y), np.asarray(proba), np.asarray(amounts, dtype=float)
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    costs = []
    for t in grid:
        flagged = proba >= t
        costs.append(amounts[(y == 1) & ~flagged].sum() + c_fp * ((y == 0) & flagged).sum())
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def total_cost(y, flagged, amounts, c_fp=COST_FP):
    y, flagged, amounts = np.asarray(y), np.asarray(flagged, dtype=bool), np.asarray(amounts, dtype=float)
    return float(amounts[(y == 1) & ~flagged].sum() + c_fp * ((y == 0) & flagged).sum())


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    train, val, test = load_split3()
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:5s}: {len(d):7d} tx | {d.card_id.nunique():5d} cartes | "
              f"fraude {d.fraud.mean():.3%} ({int(d.fraud.sum())} tx)")
    model = build_model().fit(train[BEHAVIOR_FEATURES], train["fraud"])
    pv = model.predict_proba(val[BEHAVIOR_FEATURES])[:, 1]
    thr, cost = pick_threshold(val["fraud"], pv, val["amount"])
    joblib.dump(model, MODEL)
    json.dump({"threshold": thr, "validation_cost": cost, "cost_fp": COST_FP,
               "cost_fn": "montant de la fraude non bloquée"}, open(THRESHOLD_FILE, "w"), indent=2)
    print(f"modèle -> {MODEL} | seuil (choisi sur validation) = {thr:.2f} -> {THRESHOLD_FILE}")