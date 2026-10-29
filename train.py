"""Entraînement du Random Forest de Fog A (5 features comportementales).
Pas de scaling : un Random Forest est insensible à l'échelle des features.
Split PAR CARTE (GroupShuffleSplit) : une carte n'est jamais à la fois en train et en test."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from features import BEHAVIOR_FEATURES

DATA = "data/dataset_ATM.csv"
MODEL = "artifacts/model.joblib"


def load_split(path=DATA, seed=42):
    df = pd.read_csv(path)
    missing = [c for c in BEHAVIOR_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} ne contient pas {missing} : relancer generate_dataset.py")
    df = df.dropna(subset=BEHAVIOR_FEATURES + ["fraud"])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(df, df["fraud"], groups=df["card_id"]))
    return df.iloc[tr], df.iloc[te]


def build_model():
    return RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )


if __name__ == "__main__":
    train, test = load_split()
    print(f"train={len(train)} (fraude {train.fraud.mean():.2%}) | test={len(test)} (fraude {test.fraud.mean():.2%})")
    model = build_model().fit(train[BEHAVIOR_FEATURES], train["fraud"])
    joblib.dump(model, MODEL)
    print("modèle sauvegardé ->", MODEL)
