"""CV par carte (moyenne ± écart-type), ablation de hour_of_day, split temporel J0-J30 / J30-J45,
décalage de domaine (modèle entraîné sur A, testé sur des simulateurs B aux hypothèses différentes)."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from features import BEHAVIOR_FEATURES
from generate_dataset import build
from train import build_model, load_data, load_split3, pick_threshold


def ap(model, d, feats):
    return float(average_precision_score(d["fraud"], model.predict_proba(d[feats])[:, 1]))


if __name__ == "__main__":
    df = load_data()
    out = {}
    no_hour = [f for f in BEHAVIOR_FEATURES if f != "hour_of_day"]

    cv = {"RF (9 features)": [], "RF sans hour_of_day": [], "arbre profondeur 3": []}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
    for tr, te in sgkf.split(df, df["fraud"], groups=df["card_id"]):
        a, b = df.iloc[tr], df.iloc[te]
        cv["RF (9 features)"].append(ap(build_model().fit(a[BEHAVIOR_FEATURES], a["fraud"]), b, BEHAVIOR_FEATURES))
        cv["RF sans hour_of_day"].append(ap(build_model().fit(a[no_hour], a["fraud"]), b, no_hour))
        cv["arbre profondeur 3"].append(ap(DecisionTreeClassifier(max_depth=3, random_state=0)
                                           .fit(a[BEHAVIOR_FEATURES], a["fraud"]), b, BEHAVIOR_FEATURES))
    out["cv_pr_auc_par_carte (moyenne, écart-type)"] = {
        k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for k, v in cv.items()}

    ts = pd.to_datetime(df["timestamp"])
    cut = ts.min() + pd.Timedelta(days=30)
    a, b = df[ts <= cut], df[ts > cut]
    m = build_model().fit(a[BEHAVIOR_FEATURES], a["fraud"])
    out["split_temporel (train J0-J30 / test J30-J45)"] = {
        "pr_auc": round(ap(m, b, BEHAVIOR_FEATURES), 4),
        "fraudes_test": int(b.fraud.sum()), "prevalence_test": round(float(b.fraud.mean()), 5)}

    train, val, _ = load_split3()
    mA = build_model().fit(train[BEHAVIOR_FEATURES], train["fraud"])
    thr, _ = pick_threshold(val["fraud"], mA.predict_proba(val[BEHAVIOR_FEATURES])[:, 1], val["amount"])
    shifts = {
        "A -> A (référence, test)": None,
        "B1 : sans fraude nocturne": dict(night_fraud=0.0),
        "B2 : difficulté 0.9": dict(difficulty=0.9),
        "B3 : difficulté 0.9 + sans fraude nocturne + 25 % de cartes noctambules":
            dict(difficulty=0.9, night_fraud=0.0, night_cards=0.25),
    }
    rows = {}
    for name, kw in shifts.items():
        if kw is None:
            _, _, test = load_split3()
        else:
            test = build(3000, 45, seed=7, card_rate=0.09, **{"night_cards": 0.10, "night_fraud": 0.35,
                                                              "difficulty": 0.5, **kw})
        p = mA.predict_proba(test[BEHAVIOR_FEATURES])[:, 1]
        flag = p >= thr
        y = test["fraud"].to_numpy()
        rows[name] = {"pr_auc": round(float(average_precision_score(y, p)), 4),
                      "rappel au seuil de A": round(float(flag[y == 1].mean()), 4),
                      "fp / 1000 légitimes": round(float(flag[y == 0].mean() * 1000), 2),
                      "fraudes": int(y.sum())}
    out["decalage_de_domaine (modèle entraîné sur A)"] = rows

    json.dump(out, open("artifacts/robustness.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))